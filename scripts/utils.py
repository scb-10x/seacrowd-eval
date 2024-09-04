import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

def read_all_pending_model(EVAL_REQUESTS_PATH: str) -> Dict[str, List[Tuple[Any, str]]]:
    depth = 1
    alls = defaultdict(list)
    for root, _, files in os.walk(EVAL_REQUESTS_PATH):
        current_depth = root.count(os.sep) - EVAL_REQUESTS_PATH.count(os.sep)
        if current_depth == depth:
            for file in files:
                if not file.endswith(".json"):
                    continue
                file_abs_path = os.path.join(root, file)
                with open(file_abs_path, "r") as f:
                    info = json.load(f)
                    alls[info['model']].append((info, file_abs_path))

    pendings = {}
    for k in alls.keys():
        is_pending = False
        for stat in alls[k]:
            info_dict = stat[0]
            if info_dict['status'] == "PENDING":
                is_pending = True
        if is_pending:
            pendings[k] = alls[k]
    return pendings
                
def read_model_for_name(model_name: str, EVAL_REQUESTS_PATH: str) -> List[Tuple[Any, str]]:
    depth = 1
    alls = defaultdict(list)
    for root, _, files in os.walk(EVAL_REQUESTS_PATH):
        current_depth = root.count(os.sep) - EVAL_REQUESTS_PATH.count(os.sep)
        if current_depth == depth:
            for file in files:
                if not file.endswith(".json"):
                    continue
                file_abs_path = os.path.join(root, file)
                with open(file_abs_path, "r") as f:
                    info = json.load(f)
                    alls[info['model']].append((info, file_abs_path))
    return alls[model_name]