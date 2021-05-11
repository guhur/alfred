import collections
import os
from typing import Tuple, Dict
from vocab import Vocab
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gym import Space, spaces
from models.model.vlnce.rnn_state_encoder import build_rnn_state_encoder
from models.model.vlnce.instruction_encoder import InstructionEncoder
from models.model.vlnce.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from models.model.seq2seq import Module as Base
# from models.model.seq2seq_im_mask import Module as Base
from models.utils.metric import compute_f1, compute_exact
from models.utils import get_parameter_device
from models.nn import Flatten
from gen.utils.image_util import decompress_mask


class Module(Base):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    RGB encoder
    RNN state encoder or CMA state encoder
    """

    def __init__(self, args, vocab: Dict[str, Vocab]):
        super().__init__(args, vocab)

        self.max_decode = args.max_decode
        self.teacher_forcing = True

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(args, vocab)

        observation_space = spaces.Dict(
            dict(
                rgb=spaces.Box(
                    low=0, high=0, shape=(args.pframe, args.pframe, 3), dtype=np.float32
                ),
                depth=spaces.Box(
                    low=0,
                    high=0,
                    shape=(args.pframe_depth, args.pframe_depth, 1),
                    dtype=np.float32,
                ),
            )
        )

        self.test_mode = False

        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # Init the sensors encoder
        self.rgb_encoder = TorchVisionResNet50(
            observation_space,
            args.drgb,
            spatial_output=True,
        )
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(
                1024,
                # self.rgb_encoder.output_shape[0],
                args.drgb,
            ),
            nn.ReLU(True),
        )
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=args.ddepth,
            checkpoint=args.ddppo_checkpoint,
            backbone=args.depth_backbone,
            spatial_output=True,
        )
        self.depth_linear = nn.Sequential(
            Flatten(),
            nn.Linear(
                # np.prod(self.depth_encoder.output_shape),
                66560,
                args.ddepth,
            ),
            nn.ReLU(True),
        )

        self.num_actions = len(self.vocab["action_low"])
        self._hidden_size = args.dhidden_state

        # Init the RNN state decoder
        rnn_input_size = args.ddepth
        rnn_input_size += args.drgb
        rnn_input_size += args.demb

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=args.dhidden_state,
            num_layers=1,
            rnn_type="gru",
            batch_first=True,
        )
        self.emb_action_low = nn.Embedding(self.num_actions, args.demb)


        self.rgb_kv = nn.Conv1d(
            # self.rgb_encoder.output_shape[0],
            1024,
            self._hidden_size // 2 + args.drgb,
            1,
        )

        self.depth_kv = nn.Conv1d(
            # self.depth_encoder.output_shape[0],
            65,
            self._hidden_size // 2 + args.ddepth,
            1,
        )

        self.state_q = nn.Linear(self._hidden_size, self._hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, self._hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, self._hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((self._hidden_size // 2) ** 0.5))
        )

        output_size = (
            args.dhidden_state
            + args.drgb
            + args.ddepth
            + self.instruction_encoder.output_size
             + self.emb_action_low.embedding_dim
        )
        self.second_state_compress = nn.Sequential(
            nn.Linear( output_size, self._hidden_size),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            num_layers=1,
            rnn_type="gru",
            batch_first=True,
        )
        self._output_size = args.dhidden_state

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self.action_distribution = nn.Linear(512, self.num_actions)
        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # init action
        self.go = nn.Parameter(torch.Tensor(args.demb))
        nn.init.uniform_(self.go, -0.1, 0.1)

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def enc(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:

        instruction_embedding = self.instruction_encoder(feat["lang_goal_instr"])

        if self.args.ablate_instruction:
            instruction_embedding = instruction_embedding * 0

        return instruction_embedding

    @property
    def device(self):
        return get_parameter_device(self)

    def forward(self, feat: Dict[str, torch.Tensor]):
        enc = self.enc(feat)

        # should not be there for pixel images
        frames = self.vis_dropout(feat["frames"])

        batch = enc.size(0)

        state_t = torch.zeros(
            self.num_recurrent_layers,
            batch,
            self.args.dhidden_state,
            device=self.device,
        )
        e_t = self.go.repeat(batch, 1)

        max_t = (
            feat["action_low"].size(1)
            if self.training
            else min(self.max_decode, frames.shape[1])
        )
        actions = []
        # subgoals = []
        progresses = []

        for t in range(max_t):
            obs = {
                "depth_features": torch.zeros((batch, 1, 32, 32)).to(self.device),  # TODO
                "rgb": None,  # TODO
                "rgb_features": frames[:, t],
            }
            action_t, state_t, progress_t = self.step(obs, enc, state_t, e_t)
            actions.append(action_t)
            # attn_scores.append(attn_score_t)
            # subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            w_t = (
                feat["action_low"][:, t]
                if self.teacher_forcing and self.training
                else action_t.detach().max(1)[1]
            )
            e_t = self.emb_action_low(w_t)
            
        # import ipdb
        # ipdb.set_trace()
        results = {
            "out_action_low": torch.stack(actions, dim=1),
            # "out_action_low_mask": torch.stack(masks, dim=1),
            # "out_attn_scores": torch.stack(attn_scores, dim=1),
            # "out_subgoal": torch.stack(subgoals, dim=1),
            "out_progress": torch.stack(progresses, dim=1),
            "state_t": state_t,
        }
        return results

    def step(
        self,
        observations,
        instruction_embedding,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size x T]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        first_state = rnn_hidden_states[:self.state_encoder.num_recurrent_layers]
        second_state = rnn_hidden_states[self.state_encoder.num_recurrent_layers:]

        # depth_embedding = self.depth_encoder(observations)
        # depth_embedding = torch.flatten(depth_embedding, 2)
        # FIXME
        batch = instruction_embedding.shape[0]
        device = instruction_embedding.device
        depth_embedding = torch.zeros((batch, 65, 1024), device=device) 

        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)


        # if self.args.ablate_depth:
        #     depth_embedding = depth_embedding * 0
        if self.args.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)


        # for teacher forcing, masks are directly employed on the loss
        # we keep the mask variable for future compatibility with DAgger
        masks = torch.ones_like(prev_actions[:, 0])
        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        state, first_state = self.state_encoder(state_in, second_state, masks)

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )
        text_embedding = text_state_q # DEBUG
        text_q = self.text_q(text_embedding)

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [state, text_embedding, rgb_embedding, depth_embedding, prev_actions], dim=1
        )
        x = self.second_state_compress(x)
        x, second_state = self.second_state_encoder(x, second_state, masks)

        progress_hat = torch.tanh(self.progress_monitor(x))

        logit = self.action_distribution(x)
        # progress_loss = F.mse_loss(
        
        #     progress_hat.squeeze(1), observations["progress"], reduction="none"
        # )

        rnn_hidden_states = torch.cat([first_state, second_state], dim=0)
        return logit, rnn_hidden_states, progress_hat

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # subgoal completion supervision
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            lang_goal_instr = lang_goal + lang_instr
            feat['lang_goal_instr'].append(lang_goal_instr)

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                im = torch.load(os.path.join(root, self.feat_pt))

                num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action
                num_feat_frames = im.shape[0]

                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    feat['frames'].append(im)

                # Full Dataset (contains filler frames)
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

                # low-level action mask
                # if load_mask:
                #     feat['action_low_mask'].append([self.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])

                # low-level valid interact
                feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])


        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=self.device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                #seq_lengths = np.array(list(map(len, v)))
                #embed_seq = self.emb_word(pad_seq)
                #packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                #feat[k] = packed_input
                feat[k] = pad_seq
            # elif k in {'action_low_mask'}:
            #     # mask padding
            #     seqs = [torch.tensor(vv, device=self.device, dtype=torch.float) for vv in v]
            #     feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=self.device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=self.device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                # seqs = [vv.to(device=self.device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat


    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if not self.test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask



    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow in zip(batch, out['out_action_low'].max(2)[1].tolist()):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            task_id_ann = self.get_task_and_ann_id(ex)
            pred[task_id_ann] = {
                'action_low': ' '.join(words),
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].view(-1)
        # p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        # valid_idxs = valid.view(-1).nonzero().view(-1)
        # flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0]*p_alow_mask.shape[1], *p_alow_mask.shape[2:])[valid_idxs]
        # flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)
        # alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask, flat_alow_mask)
        # losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = out['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = out['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses


    def weighted_mask_loss(self, pred_masks, gt_masks):
        '''
        mask loss that accounts for weight-imbalance between 0 and 1 pixels
        '''
        bce = self.bce_with_logits(pred_masks, gt_masks)
        flipped_mask = self.flip_tensor(gt_masks)
        inside = (bce * gt_masks).sum() / (gt_masks).sum()
        outside = (bce * flipped_mask).sum() / (flipped_mask).sum()
        return inside + outside


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}
