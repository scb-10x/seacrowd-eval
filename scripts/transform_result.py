import pandas as pd
from collections import defaultdict
from huggingface_hub import HfApi
import os
import json
from utils import read_model_for_name

HF_TOKEN = os.environ.get("TOKEN")
RESULTS_REPO = os.environ.get('RESULTS_REPO')

api = HfApi(token=HF_TOKEN)

def upload_file(result_path: str, task_type: str, model_name: str):
    api.upload_file(
        path_or_fileobj=result_path,
        path_in_repo=f'{task_type}/{model_name}/results.json',
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add {result_path} to result queue",
    )

def _get_dataset_type(dataset_name: str):
    if dataset_name in ['m3exam_tha_seacrowd_qa', "thaiexam_qa"]:
        return 'MC'
    else:
        return 'NLU'


def process_nlu_result(model_name: str, outpath: str, full_model_name=None):
    file_path = f'metrics_nlu/nlu_results_tha_{model_name}.csv'
    if not os.path.exists(file_path):
        print(f'skip upload NLU/MC result {file_path} not found')
        return False
    df = pd.read_csv(file_path)
    results = {'NLU': defaultdict(list), 'MC': defaultdict(list)}
    dataset_key = 'dataset'
    metrics_key = ['accuracy']
    for i, row in df.iterrows():
        for k in metrics_key:
            dataset_name = row[dataset_key]
            dataset_type = _get_dataset_type(dataset_name)
            results[dataset_type][dataset_name].append({k: row[k]})
    
    for task_type in results.keys():
        results_final = {}
        for k in results[task_type].keys():
            d = results[task_type][k][0]
            results_final[k] = d
            
        data = {
            'config': {
                "model_name": full_model_name if full_model_name is not None else model_name,
            },
            "results": results_final
        }

        result_path = f'{outpath}/{task_type}/{model_name}/results.json'
        print(os.path.dirname(result_path))
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as w:
            json.dump(data, w, ensure_ascii=False)
        upload_file(result_path, task_type, model_name)
    return True


def process_nlg_result(model_name: str, outpath: str, full_model_name=None):
    file_path = f'metrics_nlg/nlg_results_tha_0_{model_name}.csv'
    task_type = 'NLG'
    if not os.path.exists(file_path):
        print(f'skip upload {task_type} result {file_path} not found')
        return False
    df = pd.read_csv(file_path)
    results = defaultdict(dict)
    dataset_key = 'dataset'
    dataset_to_metrics = {
        'xl_sum_tha_seacrowd_t2t': ['ROUGE1', 'ROUGE2', 'ROUGEL'],
        'flores200_eng_Latn_tha_Thai_seacrowd_t2t': ['BLEU', 'SacreBLEU', 'chrF++'],
        'flores200_tha_Thai_eng_Latn_seacrowd_t2t': ['BLEU', 'SacreBLEU', 'chrF++'],
        'iapp_squad_seacrowd_qa': ['ROUGE1', 'ROUGE2', 'ROUGEL']
    }

    for i, row in df.iterrows():
        metrics_key = dataset_to_metrics[row[dataset_key]]
        for k in metrics_key:
            results[row[dataset_key]][k] = row[k]
    
    data = {
        'config': {
            "model_name": full_model_name if full_model_name is not None else model_name,
        },
        "results": results
    }
    result_path = f'{outpath}/{task_type}/{model_name}/results.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as w:
        json.dump(data, w, ensure_ascii=False)
    
    upload_file(result_path, task_type, model_name)
    return True

def process_llm_result(model_name: str, outpath: str, full_model_name=None):
    task_type = 'LLM'
    file_path = f'metrics_llm/{model_name}.json'
    if not os.path.exists(file_path):
        print(f'skip upload {task_type} result {file_path} not found')
        return False
    results = defaultdict(dict)
    with open(file_path) as f:
        d = json.load(f)
        for m in d.keys():
            for name in d[m].keys():
                results[f'{name}'][m] = d[m][name]
    data = {
        'config': {
            "model_name": full_model_name if full_model_name is not None else model_name,
        },
        "results": results
    }
    result_path = f'{outpath}/{task_type}/{model_name}/results.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as w:
        json.dump(data, w, ensure_ascii=False)
        
    upload_file(result_path, task_type, model_name)
    return True
    

def update_model_status(full_model_name: str):
    try:
        CACHE_PATH = os.getenv("HF_HOME", ".")
        EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
        all_models_with_names = read_model_for_name(full_model_name, EVAL_REQUESTS_PATH)
        QUEUE_REPO = os.environ["QUEUE_REPO"]
        for v, p in all_models_with_names:
            v['status'] = 'FINISHED'
            with open(p, 'w') as w:
                json.dump(v, w, ensure_ascii=False)
            api.upload_file(
                path_or_fileobj=p,
                path_in_repo=p.split("eval-queue/")[1],
                repo_id=QUEUE_REPO,
                repo_type="dataset",
                commit_message=f"Update {model_name} to eval queue",
            )
    except Exception as e:
        print(e)
        pass
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='LLM as judge evalulator')
    parser.add_argument('model_name')
    parser.add_argument('--result_path', default='results')
    args = parser.parse_args()
    full_model_name = args.model_name
    if '/' in full_model_name:
        model_name = full_model_name.split('/')[-1]
    else:
        model_name = full_model_name
        full_model_name = f'api/{model_name}'
    has_success = False
    status = process_nlu_result(model_name, args.result_path, full_model_name=full_model_name)
    if status:
        has_success = True
    status = process_nlg_result(model_name, args.result_path, full_model_name=full_model_name)
    if status:
        has_success = True
    status = process_llm_result(model_name, args.result_path, full_model_name=full_model_name)
    if status:
        has_success = True
    if has_success:
        update_model_status(full_model_name)