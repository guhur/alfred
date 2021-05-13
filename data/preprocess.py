import os
import json
from typing import Dict, Tuple
from pathlib import Path
import revtok
import torch
import copy
import progressbar
from vocab import Vocab
from gen.utils.py_util import remove_spaces_and_lower
from gen.utils.game_util import sample_templated_task_desc_from_traj_data
from utils.misc import load_json, save_json


class Dataset(object):
    def __init__(
        self,
        data: Path,
        frame_size: Tuple[int, int],
        use_templated_goals: bool,
    ):
        self.data_dir = data
        self.frame_size = frame_size
        self.use_templated_goals = use_templated_goals

        self.vocab = {
            "word": Vocab(["<<pad>>", "<<seg>>", "<<goal>>"]),
            "action_low": Vocab(["<<pad>>", "<<seg>>", "<<stop>>"]),
            "action_high": Vocab(["<<pad>>", "<<seg>>", "<<stop>>"]),
        }
        self.word_seg = self.vocab["word"].word2index("<<seg>>", train=False)
        self.non_interact_actions = (
            "MoveAhead",
            "Rotate",
            "Look",
            "<<stop>>",
            "<<pad>>",
            "<<seg>>",
        )

    @staticmethod
    def numericalize(vocab, words, train=True):
        """
        converts words to unique integers
        """
        return vocab.word2index([w.strip().lower() for w in words], train=train)

    def preprocess_splits(self, splits):
        """
        saves preprocessed data as jsons in specified folder
        """
        for k, d in splits.items():
            print("Preprocessing {}".format(k))
            train_mode = "test" not in k

            for task in progressbar.progressbar(d):
                task_dir = self.data_dir / k / task["task"]

                ex = load_json(task_dir / "traj_data.json")

                # copy trajectory
                # repeat_idx is the index of the annotation for each trajectory
                r_idx = task["repeat_idx"]
                traj = ex.copy()

                # root & split
                traj["root"] = task_dir / task["task"]
                traj["task"] = task["task"]
                traj["split"] = k
                traj["repeat_idx"] = r_idx

                # numericalize language
                # templated goals are not available for the test set
                use_templated_goals = self.use_templated_goals and train_mode
                self.process_language(
                    ex, traj, r_idx, use_templated_goals=use_templated_goals
                )

                # numericalize actions for train/valid splits
                # expert actions are not available for the test set
                if train_mode:
                    self.process_actions(ex, traj)

                # check if preprocessing storage folder exists
                preprocessed_folder = task_dir / "preprocessing"
                preprocessed_folder.mkdir(exist_ok=True)

                # save preprocessed json
                save_json(traj, preprocessed_folder / f"ann_{r_idx}.json")

        # save vocab in data path
        print("Word voc size", len(self.vocab["word"]))
        print("Number of actions", len(self.vocab["action_low"]))
        vocab_data_path = os.path.join(self.data_dir, "preprocess.vocab")
        torch.save(self.vocab, vocab_data_path)

    def process_language(self, ex, traj, r_idx, use_templated_goals=False):
        # goal instruction
        if use_templated_goals:
            task_desc = sample_templated_task_desc_from_traj_data(traj)
        else:
            task_desc = ex["turk_annotations"]["anns"][r_idx]["task_desc"]

        # step-by-step instructions
        high_descs = ex["turk_annotations"]["anns"][r_idx]["high_descs"]

        # tokenize language
        traj["ann"] = {
            "goal": revtok.tokenize(remove_spaces_and_lower(task_desc)) + ["<<goal>>"],
            "instr": [revtok.tokenize(remove_spaces_and_lower(x)) for x in high_descs]
            + [["<<stop>>"]],
            "repeat_idx": r_idx,
        }

        # numericalize language
        traj["num"] = {}
        traj["num"]["lang_goal"] = self.numericalize(
            self.vocab["word"], traj["ann"]["goal"], train=True
        )
        traj["num"]["lang_instr"] = [
            self.numericalize(self.vocab["word"], x, train=True)
            for x in traj["ann"]["instr"]
        ]

    def process_actions(self, ex, traj):
        # deal with missing end high-level action
        self.fix_missing_high_pddl_end_action(ex)

        # end action for low_actions
        end_action = {
            "api_action": {"action": "NoOp"},
            "discrete_action": {"action": "<<stop>>", "args": {}},
            "high_idx": ex["plan"]["high_pddl"][-1]["high_idx"],
        }

        # init action_low and action_high
        num_hl_actions = len(ex["plan"]["high_pddl"])
        traj["num"]["action_low"] = [
            list() for _ in range(num_hl_actions)
        ]  # temporally aligned with HL actions
        traj["num"]["action_high"] = []
        low_to_high_idx = []

        for a in ex["plan"]["low_actions"] + [end_action]:
            # high-level action index (subgoals)
            high_idx = a["high_idx"]
            low_to_high_idx.append(high_idx)

            # low-level action (API commands)
            traj["num"]["action_low"][high_idx].append(
                {
                    "high_idx": a["high_idx"],
                    "action": self.vocab["action_low"].word2index(
                        a["discrete_action"]["action"], train=True
                    ),
                    "action_high_args": a["discrete_action"]["args"],
                }
            )

            # low-level bounding box (not used in the model)
            if "bbox" in a["discrete_action"]["args"]:
                xmin, ymin, xmax, ymax = [
                    float(x) if x != "NULL" else -1
                    for x in a["discrete_action"]["args"]["bbox"]
                ]
                traj["num"]["action_low"][high_idx][-1]["centroid"] = [
                    (xmin + (xmax - xmin) / 2) / self.frame_size[1],
                    (ymin + (ymax - ymin) / 2) / self.frame_size[0],
                ]
            else:
                traj["num"]["action_low"][high_idx][-1]["centroid"] = [-1, -1]

            # low-level interaction mask (Note: this mask needs to be decompressed)
            if "mask" in a["discrete_action"]["args"]:
                mask = a["discrete_action"]["args"]["mask"]
            else:
                mask = None
            traj["num"]["action_low"][high_idx][-1]["mask"] = mask

            # interaction validity
            traj["num"]["action_low"][high_idx][-1]["valid_interact"] = int(
                a["discrete_action"]["action"] not in self.non_interact_actions
            )

        # low to high idx
        traj["num"]["low_to_high_idx"] = low_to_high_idx

        # high-level actions
        for a in ex["plan"]["high_pddl"]:
            traj["num"]["action_high"].append(
                {
                    "high_idx": a["high_idx"],
                    "action": self.vocab["action_high"].word2index(
                        a["discrete_action"]["action"], train=True
                    ),
                    "action_high_args": self.numericalize(
                        self.vocab["action_high"], a["discrete_action"]["args"]
                    ),
                }
            )

        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = len(traj["num"]["action_low"])
        lang_instr_seg_len = len(traj["num"]["lang_instr"])
        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 0:
            assert seg_len_diff == 1  # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            self.merge_last_two_low_actions(traj)

    def fix_missing_high_pddl_end_action(self, ex):
        """
        appends a terminal action to a sequence of high-level actions
        """
        if ex["plan"]["high_pddl"][-1]["planner_action"]["action"] != "End":
            ex["plan"]["high_pddl"].append(
                {
                    "discrete_action": {"action": "NoOp", "args": []},
                    "planner_action": {"value": 1, "action": "End"},
                    "high_idx": len(ex["plan"]["high_pddl"]),
                }
            )

    def merge_last_two_low_actions(self, conv):
        """
        combines the last two action sequences into one sequence
        """
        extra_seg = copy.deepcopy(conv["num"]["action_low"][-2])
        for sub in extra_seg:
            sub["high_idx"] = conv["num"]["action_low"][-3][0]["high_idx"]
            conv["num"]["action_low"][-3].append(sub)
        del conv["num"]["action_low"][-2]
        conv["num"]["action_low"][-1][0]["high_idx"] = (
            len(conv["plan"]["high_pddl"]) - 1
        )

