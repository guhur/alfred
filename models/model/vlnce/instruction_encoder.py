import gzip
import json
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import argtyped


class InstructionEncoderArguments(argtyped.Arguments):
    language_hidden_size: int = 128
    voc_size: int = 2360
    language_dropout: float = 0.1
    language_embedding_size: int = 50
    use_pretrained_embeddings: bool = True
    fine_tune_embeddings: bool = False
    embedding_file: Path = Path(
        "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
    )


class InstructionEncoder(nn.Module):
    def __init__(self, args: InstructionEncoderArguments):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.
        """
        super().__init__()

        self.config = args

        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze=not self.config.fine_tune_embeddings,
            )
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=args.voc_size,
                embedding_dim=self.config.language_embedding_size,
                padding_idx=0,
            )

        rnn = nn.GRU
        self.bidir = True
        self.encoder_rnn = rnn(
            input_size=self.config.language_embedding_size,
            hidden_size=self.config.language_hidden_size,
            bidirectional=self.bidir,
        )

    @property
    def output_size(self):
        return self.config.language_hidden_size * (2 if self.bidir else 1)

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
