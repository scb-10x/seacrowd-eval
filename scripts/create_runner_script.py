import json
from huggingface_hub import HfApi, snapshot_download
from utils import read_all_pending_model
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    api = HfApi()
    outpath = "./"
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
    alls = read_all_pending_model(EVAL_REQUESTS_PATH)
    to_run = []
    for model_name in alls.keys():
        for request_row in alls[model_name]:
            info, filepath = request_row
            status = info["status"]
            model_name = info["model"]
            if status == "PENDING":
                to_run.append(model_name)
                info['status'] = 'RUNNING'
                with open(filepath, 'w') as w:
                    json.dump(info, w, ensure_ascii=False)
            # TODO / turn-on-this: Update status
            #     api.upload_file(
            #         path_or_fileobj=filepath,
            #         path_in_repo=filepath.split("eval-queue/")[1],
            #         repo_id=QUEUE_REPO,
            #         repo_type="dataset",
            #         commit_message=f"Update {model_name} to eval queue",
            #     )
    
    to_run = list(set(to_run))
    with open(f"{outpath}/run.sh", "w") as w:
        for model_name in to_run:
            w.write(f"MODEL_NAME={model_name} sh runner.sh\n")


if __name__ == "__main__":
    main()
