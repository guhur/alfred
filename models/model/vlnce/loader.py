import collections
import os
from typing import Tuple, Dict, Iterable, List, Callable, Any, Optional
from typing_extensions import TypedDict
import json
from pathlib import Path
import pprint
from vocab import Vocab
import argtyped
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gym import Space, spaces
import pytorch_lightning as pl

# from models.model.seq2seq_im_mask import Module as Base
from models.utils.metric import compute_f1, compute_exact
from models.utils import get_parameter_device
from models.nn import Flatten
from gen.utils.image_util import decompress_mask
from data.preprocess import Dataset as PreprocessDataset
from utils.misc import load_json


Task = TypedDict("Task", {"repeat_idx": int, "task": str})


class VLNCEDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        tasks: List[Task],
        max_subgoals: int,
        ablate_goal: bool,
        ablate_instr: bool,
    ):
        self.tasks = tasks
        self.split = split
        self.data_dir = data_dir
        self.max_subgoals = max_subgoals
        self.ablate_goal = ablate_goal
        self.ablate_instr = ablate_instr
        self.test_mode = "test" in split
        self.feat_pt = "feat_conv.pt"

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # TODO rewrite the preprocessing to build independant train files for each experiment
        #       this getitem should be much smaller

    def _load_task_json(self, task: Task):
        """
        load preprocessed json from disk
        """
        json_path = (
            self.data_dir
            / self.split
            / task["task"]
            / "preprocessing"
            / f"ann_{task['repeat_idx']}.json"
        )
        return load_json(json_path)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        task = self.tasks[index]
        ex = self._load_task_json(task)
        feat: Dict[str, Any] = {}

        # subgoal completion supervision
        if "low_to_high_idx" in ex["num"]:
            feat["subgoals_completed"] = (
                np.array(ex["num"]["low_to_high_idx"]) / self.max_subgoals
            )

        # progress monitor supervision
        if "action_low" in ex["num"]:
            num_actions = len([a for sg in ex["num"]["action_low"] for a in sg])
            subgoal_progress = [
                (i + 1) / float(num_actions) for i in range(num_actions)
            ]
            feat["subgoal_progress"] = subgoal_progress

        # serialize segments
        self.serialize_lang_action(ex)

        # goal and instr language
        lang_goal, lang_instr = ex["num"]["lang_goal"], ex["num"]["lang_instr"]

        # zero inputs if specified
        lang_goal = self._zero_input(lang_goal) if self.ablate_goal else lang_goal
        lang_instr = self._zero_input(lang_instr) if self.ablate_instr else lang_instr

        # append goal + instr
        feat["lang_goal_instr"] = lang_goal + lang_instr

        # load Resnet features from disk
        if not self.test_mode:
            root = self._get_task_root(ex)
            im = torch.load(os.path.join(root, self.feat_pt))

            num_low_actions = (
                len(ex["plan"]["low_actions"]) + 1
            )  # +1 for additional stop action
            num_feat_frames = im.shape[0]

            # Modeling Quickstart (without filler frames)
            if num_low_actions == num_feat_frames:
                feat["frames"] = im

            # Full Dataset (contains filler frames)
            else:
                keep = [None] * num_low_actions
                for i, d in enumerate(ex["images"]):
                    # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                    if keep[d["low_idx"]] is None:
                        keep[d["low_idx"]] = im[i]
                keep[-1] = im[-1]  # stop frame
                feat["frames"] = torch.stack(keep, dim=0)

        #########
        # outputs
        #########

        if not self.test_mode:
            # low-level action
            feat["action_low"] = [a["action"] for a in ex["num"]["action_low"]]

            # low-level action mask
            # if load_mask:
            #     feat['action_low_mask'].append([self.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])

            # low-level valid interact
            feat["action_low_valid_interact"] = [
                a["valid_interact"] for a in ex["num"]["action_low"]
            ]

        return feat

    def _get_task_root(self, ex):
        """
        returns the folder path of a trajectory
        """
        return self.data_dir / self.split / ex["task"]

    def serialize_lang_action(self, feat):
        """
        append segmented instr language and low-level actions into single sequences
        """
        is_serialized = not isinstance(feat["num"]["lang_instr"][0], list)
        if not is_serialized:
            feat["num"]["lang_instr"] = [
                word for desc in feat["num"]["lang_instr"] for word in desc
            ]
            if not self.test_mode:
                feat["num"]["action_low"] = [
                    a for a_group in feat["num"]["action_low"] for a in a_group
                ]

    def _zero_input(self, x, keep_end_token=True):
        """
        pad input with zeros (used for ablations)
        """
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def _zero_input_list(self, x, keep_end_token=True):
        """
        pad a list of input with zeros (used for ablations)
        """
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    def __len__(self):
        return len(self.tasks)


class DataModuleArguments(argtyped.Arguments):
    data: Path = Path("data/json_feat_2.1.0")
    batch_size: int = 32
    use_templated_goals: bool = False
    split_file: Path = Path("splits/oct21.json")
    pframe: int = 224
    max_subgoals: int = 25
    num_workers: int = 4


class DataModule(pl.LightningDataModule):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    RGB encoder
    RNN state encoder or CMA state encoder
    """

    def prepare_data(self):
        if self.vocab_path.exists():
            return
        dataset = PreprocessDataset(
            self.args.data,
            (self.args.pframe, self.args.pframe),
            self.args.use_templated_goals,
        )
        dataset.preprocess_splits(self.split_tasks)

    def __init__(
        self,
        args: DataModuleArguments,
    ):
        super().__init__()
        self.args = args
        self.split_tasks = load_json(args.split_file)
        self.vocab_path = self.args.data / "preprocess.vocab"

        # sentinel tokens
        self.pad = 0
        self.seg = 1

    def setup(self, stage: Optional[str] = None):
        self.vocab: Dict[str, Vocab] = torch.load(self.vocab_path)
        # end tokens
        self.stop_token = self.vocab["action_low"].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab["action_low"].word2index("<<seg>>", train=False)

    def _create_dataset(self, split: str) -> Dataset:
        return VLNCEDataset(
            self.args.data,
            split,
            self.split_tasks[split],
            self.args.max_subgoals,
            False,
            False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._create_dataset("train"),
            batch_size=self.args.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                self._create_dataset("valid_seen"),
                batch_size=self.args.batch_size,
                collate_fn=self._collate_fn,
                num_workers=self.args.num_workers,
            ),
            DataLoader(
                self._create_dataset("valid_unseen"),
                batch_size=self.args.batch_size,
                collate_fn=self._collate_fn,
                num_workers=self.args.num_workers,
            ),
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return []

    #           DataLoader(
    #               self._create_dataset("test_seen"),
    #               batch_size=self.batch_size,
    #               collate_fn=self._collate_fn,
    #           ),
    #           DataLoader(
    #               self._create_dataset("test_unseen"),
    #               batch_size=self.batch_size,
    #               collate_fn=self._collate_fn,
    #           ),
    #       ]

    def _collate_fn(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}

        # tensorization and padding
        for k in samples[0].keys():
            if k in {"lang_goal_instr"}:
                # language embedding and padding
                seqs = [torch.tensor(feat[k]) for feat in samples]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                # seq_lengths = np.array(list(map(len, v)))
                # embed_seq = self.emb_word(pad_seq)
                # packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                # feat[k] = packed_input
                batch[k] = pad_seq
            # elif k in {'action_low_mask'}:
            #     # mask padding
            #     seqs = [torch.tensor(vv, device=self.device, dtype=torch.float) for vv in v]
            #     feat[k] = seqs
            elif k in {"subgoal_progress", "subgoals_completed"}:
                # auxillary padding
                seqs = [torch.tensor(feat[k], dtype=torch.float) for feat in samples]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                batch[k] = pad_seq
            else:
                # default: tensorize and pad sequence
                dtype = torch.float if ("frames" in k) else torch.long
                seqs = [torch.tensor(feat[k], dtype=dtype) for feat in samples]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                batch[k] = pad_seq

        return batch
