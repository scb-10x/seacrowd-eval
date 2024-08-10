import os, sys
import csv
from os.path import exists

import pandas as pd
from tqdm import tqdm
from metrics_utils import generation_metrics_fn
from model_utils import load_model_and_tokenizer
from prompt_utils import get_prompt, get_lang_name
from data_utils import load_nlg_datasets

import torch
from transformers import set_seed
from nusacrowd.utils.constants import Tasks
from anyascii import anyascii


def to_prompt(input, prompt, prompt_lang, task_name, task_type, with_label=False):
    if '[INPUT]' in prompt:
        prompt = prompt.replace('[INPUT]', input['text_1'].strip())

    if task_type == Tasks.MACHINE_TRANSLATION.value:

        # Extract src and tgt based on nusantara config name
        task_names = task_name.split('_')

        if "flores200" in task_name:
            src_lang = task_names[-6]
            tgt_lang = task_names[-4]

        else:
            src_lang = task_names[-4]
            tgt_lang = task_names[-3]

        # Replace src and tgt lang name
        prompt = prompt.replace('[SOURCE]', get_lang_name(prompt_lang, src_lang))
        prompt = prompt.replace('[TARGET]', get_lang_name(prompt_lang, tgt_lang))
    
    if task_type == Tasks.QUESTION_ANSWERING.value:
        prompt = prompt.replace('[CONTEXT]', input['context'].strip())
        prompt = prompt.replace('[QUESTION]', input['question'].strip())
        # remove line that mention about [LABEL_CHOICE]
        new_prompts = []
        for p in prompt.split('\n'):
            if '[ANSWER_CHOICES]' not in p:
                new_prompts.append(p)
        prompt = '\n'.join(new_prompts)
        prompt = prompt.replace('[LABEL_CHOICE]', '')

    if with_label:
        if task_type == Tasks.QUESTION_ANSWERING.value:
            prompt += " " + input['answer'][0]
        else:
            prompt += " " + input['text_2']
    
    return prompt

def _get_terminator(tokenizer):
    eos_tokens = ["<|eot_id|>", "<|im_start|>", "<|im_end|>"]
    terminators = [
        tokenizer.eos_token_id,
    ]
    for t in eos_tokens:
        tok = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tok, int):
            terminators.append(tok)
    return terminators

def predict_generation(prompts, model_name, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    input_size = inputs["input_ids"].shape[1]
    if 'sea-lion' in model_name and 'token_type_ids' in inputs.keys():
        del inputs["token_type_ids"]

    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=300, eos_token_id=_get_terminator(tokenizer))
    preds = tokenizer.batch_decode(outputs[:,input_size:], skip_special_tokens=True)
    return preds


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise ValueError('main_nlg_prompt.py <prompt_lang> <model_path_or_name> <n_shot> <n_batch>')

    out_dir = './outputs_nlg'
    metric_dir = './metrics_nlg'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    prompt_lang = sys.argv[1]
    MODEL = sys.argv[2]
    N_SHOT = int(sys.argv[3])
    N_BATCH = int(sys.argv[4])
    SAVE_EVERY = 10

    # Load prompt
    prompt_templates = get_prompt(prompt_lang, return_only_one=True)

    # Load Dataset
    print('Load NLG Datasets...')
    nlg_datasets = load_nlg_datasets()

    print(f'Loaded {len(nlg_datasets)} NLG datasets')
    for i, dset_subset in enumerate(nlg_datasets.keys()):
        print(f'{i} {dset_subset}')

    # Set seed
    set_seed(42)

    # Load Model & Tokenizer
    # Tokenizer initialization
    model, tokenizer = load_model_and_tokenizer(MODEL, compile=True)

    metrics = {'dataset': []}
    for i, dset_subset in enumerate(nlg_datasets.keys()):
        nlg_dset, task_type = nlg_datasets[dset_subset]

        print(f"{i} {dset_subset} {task_type}")
        
        if task_type.value not in prompt_templates or nlg_dset is None:
            continue

        if 'test' in nlg_dset.keys():
            data = nlg_dset['test']
        elif 'validation' in nlg_dset.keys():
            data = nlg_dset['validation']
        elif 'devtest' in nlg_dset.keys():
            data = nlg_dset['devtest']
        else:
            data = nlg_dset['train']

        if 'train' in nlg_dset.keys():
            few_shot_data = nlg_dset['train']
        elif 'devtest' in nlg_dset.keys():
            few_shot_data = nlg_dset['devtest']
        elif 'test' in nlg_dset.keys():
            few_shot_data = nlg_dset['test']

        for prompt_id, prompt_template in enumerate(prompt_templates[task_type.value]):
            inputs = []
            preds = []
            preds_latin = []
            golds = []
            print(f"PROMPT ID: {prompt_id}")
            print(f"SAMPLE PROMPT: {to_prompt(data[0], prompt_template, prompt_lang, dset_subset, task_type.value)}")

            few_shot_text_list = []
            if N_SHOT > 0:
                for sample in tqdm(few_shot_data):
                    # Skip shot examples
                    if task_type != Tasks.QUESTION_ANSWERING and len(sample['text_1']) < 20:
                        continue
                    few_shot_text_list.append(
                        to_prompt(sample, prompt_template, prompt_lang, dset_subset, task_type.value, with_label=True)
                    )
                    if len(few_shot_text_list) == N_SHOT:
                        break
            print(f'FEW SHOT SAMPLES: {few_shot_text_list}')
            
            # Zero-shot inference
            if exists(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv'):        
                print("Output exist, use existing log instead")
                with open(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        preds_latin.append(row["Pred_Latin"])
                        golds.append(row["Gold"])
                print(f"Skipping until {len(preds)}")

            # If incomplete, continue
            if len(preds) < len(data):
                count = 0
                with torch.inference_mode():
                    prompts, batch_golds = [], []
                    for e, sample in enumerate(tqdm(data)):
                        if e < len(preds):
                            continue
                        
                        # Buffer
                        prompt_text = to_prompt(sample, prompt_template, prompt_lang, dset_subset, task_type.value)
                        prompt_text = '\n\n'.join(few_shot_text_list + [prompt_text])
                        prompt_text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_text}], add_generation_prompt=True, tokenize=False)
                        prompts.append(prompt_text)

                        batch_golds.append(sample['answer'][0] if 'answer' in sample else sample['text_2'])

                        # Batch inference
                        if len(prompts) == N_BATCH:
                            batch_preds = predict_generation(prompts, MODEL, tokenizer, model)
                            for (prompt_text, pred, gold) in zip(prompts, batch_preds, batch_golds):
                                inputs.append(prompt_text)
                                preds.append(pred if pred is not None else '')
                                preds_latin.append(anyascii(pred) if pred is not None else '')
                                golds.append(gold)
                            prompts, batch_golds = [], []
                            count += 1

                        if count == SAVE_EVERY:
                            # partial saving
                            inference_df = pd.DataFrame(list(zip(inputs, preds, preds_latin, golds)), columns=['Input', 'Pred', 'Pred_Latin', 'Gold'])
                            inference_df.to_csv(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)
                            count = 0

                    # Predict the rest inputs
                    if len(prompts) > 0:
                        batch_preds = predict_generation(prompts, MODEL, tokenizer, model)
                        for (prompt_text, pred, gold) in zip(prompts, batch_preds, batch_golds):
                            inputs.append(prompt_text)
                            preds.append(pred if pred is not None else '')
                            preds_latin.append(anyascii(pred) if pred is not None else '')
                            golds.append(gold)
                        prompts, batch_golds = [], []
            
            # Final save
            inference_df = pd.DataFrame(list(zip(inputs, preds, preds_latin, golds)), columns=['Input', 'Pred', 'Pred_Latin', 'Gold'])
            inference_df.to_csv(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)

            # To accomodate old bug where list are not properly re-initiated
            inputs = inputs[-len(data):]
            preds = preds[-len(data):]
            preds_latin = preds_latin[-len(data):]
            golds = golds[-len(data):]

            eval_metric = generation_metrics_fn(preds, golds)
            eval_metric_latin = generation_metrics_fn(preds_latin, golds)
            for key, value in eval_metric_latin.items():
                eval_metric[f'{key}_latin'] = value

            print(f'== {dset_subset} == ')
            for k, v in eval_metric.items():
                print(k, v)            
            print("===\n\n")
            eval_metric['prompt_id'] = prompt_id

            metrics['dataset'].append(dset_subset)
            for k in eval_metric:
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(eval_metric[k])


    pd.DataFrame.from_dict(metrics).reset_index().to_csv(f'{metric_dir}/nlg_results_{prompt_lang}_{N_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)
