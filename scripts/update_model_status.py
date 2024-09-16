import json
import os
from typing import List
from huggingface_hub import HfApi, snapshot_download
from utils import read_model_for_name

def main(model_names: List[str], status: str):
    api = HfApi()
    CACHE_PATH = os.getenv("HF_HOME", ".")
    EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
    QUEUE_REPO = os.environ["QUEUE_REPO"]
    snapshot_download(
        repo_id=QUEUE_REPO,
        local_dir=EVAL_REQUESTS_PATH,
        repo_type="dataset",
        tqdm_class=None,
        etag_timeout=30,
    )
    assert status in ['FINISHED', 'FAILED', 'PENDING']
    EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
    for full_model_name in model_names:
        all_models_with_names = read_model_for_name(full_model_name, EVAL_REQUESTS_PATH)
        for v, p in all_models_with_names:
            v['status'] = status
            with open(p, 'w') as w:
                json.dump(v, w, ensure_ascii=False)
            print(f'uploading: {p}')
            api.upload_file(
                path_or_fileobj=p,
                path_in_repo=p.split("eval-queue/")[1],
                repo_id=QUEUE_REPO,
                repo_type="dataset",
                commit_message=f"Update {p} to eval queue",
            )

if __name__ == '__main__':
    main(['...'], status='PENDING')