import json
import argparse
from typing import Union
from pathlib import Path


PathOrStr = Union[Path, str]


def load_json(filename: PathOrStr):
    with open(filename) as fid:
        return json.load(fid)


def save_json(data, filename: PathOrStr):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_task_and_ann_id(ex):
    """
    single string for task_id and annotation repeat idx
    """
    return "%s_%s" % (ex["task_id"], str(ex["repeat_idx"]))
