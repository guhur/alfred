import gzip
import json
from typing import Dict
import torch
import torch.nn as nn
from vocab import Vocab


class InstructionEncoder(nn.Module):
    def __init__(self, args, vocab: Dict[str, Vocab]):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.
        """
        super().__init__()

        self.config = args
        self.vocab = vocab

        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze=not self.config.fine_tune_embeddings,
            )
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=len(self.vocab["word"]),
                embedding_dim=self.config.demb,
                padding_idx=0,
            )

        rnn = nn.GRU
        self.bidir = True
        self.encoder_rnn = rnn(
            input_size=self.config.demb,
            hidden_size=self.config.dhidden_instr,
            bidirectional=self.bidir,
        )

    @property
    def output_size(self):
        return self.config.dhidden_instr * (2 if self.bidir else 1)

    def _load_embeddings(self):
        """Loads word embeddings from a pretrained embeddings file.

        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged:
            https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ

        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, instruction):
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = instruction.long()

        lengths = (instruction != 0.0).long().sum(dim=1)
        embedded = self.embedding_layer(instruction)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu().long(), batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0].permute(
            0, 2, 1
        )
