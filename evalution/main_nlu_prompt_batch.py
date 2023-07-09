"""nusacrowd zero-shot prompt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ru8DyS2ALWfRdkjOPHj-KNjw6Pfa44Nd
"""
import os, sys
import csv
from os.path import exists

from numpy import argmax, stack
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from nusacrowd import NusantaraConfigHelper
from nusacrowd.utils.constants import Tasks

from prompt_utils import get_prompt
from data_utils import load_nlu_datasets

#!pip install git+https://github.com/IndoNLP/nusa-crowd.git@release_exp
#!pip install transformers
#!pip install sentencepiece

DEBUG=False

def to_prompt(input, prompt, labels, prompt_lang):
    # single label
    if 'text' in input:
        prompt = prompt.replace('[INPUT]', input['text'])
    else:
        prompt = prompt.replace('[INPUT_A]', input['text_1'])
        prompt = prompt.replace('[INPUT_B]', input['text_2'])

    # replace [OPTIONS] to A, B, or C
    if "[OPTIONS]" in prompt:
        new_labels = [f'{l}' for l in labels]
        new_labels[-1] = ("or " if 'EN' in prompt_lang else  "atau ") + new_labels[-1] 
        if len(new_labels) > 2:
            prompt = prompt.replace('[OPTIONS]', ', '.join(new_labels))
        else:
            prompt = prompt.replace('[OPTIONS]', ' '.join(new_labels))

    return prompt


@torch.inference_mode()
def get_logprobs(model, tokenizer, inputs, label_ids=None, label_attn=None):
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')
    input_ids, output_ids, attn_mask = inputs["input_ids"][:,:-1], inputs["input_ids"][:, 1:], inputs['attention_mask'][:,:-1]
    
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=output_ids)
    logits = outputs.logits
    
    if model.config.is_encoder_decoder:
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, label_ids.unsqueeze(2)).squeeze(dim=-1) * label_attn # Zero-out Padding Token
        return logprobs.squeeze(dim=-1).sum(dim=-1).cpu().detach().numpy()
    else:
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)).squeeze(dim=-1)
        logprobs[input_ids == tokenizer.pad_token_id] = 0
        return logprobs.squeeze(dim=-1).sum(dim=1).cpu().detach().numpy()

@torch.inference_mode()
def predict_classification(model, tokenizer, prompts, labels):
    if model.config.is_encoder_decoder:
        labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
        list_label_ids = labels_encoded['input_ids'].to('cuda')
        list_label_attn = labels_encoded['attention_mask'].to('cuda')
        
        inputs = [prompt.replace('[LABELS_CHOICE]', '') for prompt in prompts]
        probs = []
        for (label_ids, label_attn) in zip(list_label_ids, list_label_attn):
            probs.append(
                get_logprobs(model, tokenizer, inputs, label_ids.view(1,-1), label_attn.view(1,-1))
            )
    else:
        probs = []
        for label in labels:
            inputs = []
            for prompt in prompts:
                inputs.append(prompt.replace('[LABELS_CHOICE]', label))
            probs.append(get_logprobs(model, tokenizer, inputs))
    return probs

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise ValueError('main_nlu_prompt.py <prompt_lang> <model_path_or_name> <batch_size> <save_every (OPTIONAL)>')

    prompt_lang = sys.argv[1]
    MODEL = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])
    
    SAVE_EVERY = 10
    if len(sys.argv) == 5:
        SAVE_EVERY = int(sys.argv[4])

    os.makedirs('./outputs_nlu', exist_ok=True) 
    os.makedirs('./metrics', exist_ok=True) 

    # Load Prompt
    TASK_TYPE_TO_PROMPT = get_prompt(prompt_lang)

    # Load Dataset
    print('Load NLU Datasets...')
    nlu_datasets = load_nlu_datasets()

    print(f'Loaded {len(nlu_datasets)} NLU datasets')
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, truncation_side='left', padding_side='right')
    if "bloom" in MODEL or "xglm" in MODEL or "gpt2" in MODEL:
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
        tokenizer.pad_token = tokenizer.eos_token # Use EOS to pad label
        
    model.eval()
    torch.no_grad()

    metrics = {}
    labels = []
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')
        nlu_dset, task_type = nlu_datasets[dset_subset]
        print(TASK_TYPE_TO_PROMPT.keys())
        if task_type.value not in TASK_TYPE_TO_PROMPT:
            print('SKIP')
            continue

        # Retrieve metadata
        split = 'test'
        if 'test' in nlu_dset.keys():
            test_dset = nlu_dset['test']
        else:
            test_dset = nlu_dset['train']
            split = 'train'
        print(f'Processing {dset_subset}')

        # Retrieve & preprocess labels
        try:
            label_names = test_dset.features['label'].names
        except:
            label_names = list(set(test_dset['label']))
        label_names = [str(label).lower().replace("_"," ") for label in label_names]
        label_to_id_dict = { l : i for i, l in enumerate(label_names)}
            
        for prompt_id, prompt_template in enumerate(TASK_TYPE_TO_PROMPT[task_type.value]):
            inputs, preds, golds = [], [], []
            
            # Check saved data
            if exists(f'outputs_nlu/{dset_subset}_{prompt_id}_{MODEL.split("/")[-1]}.csv'):
                print("Output exist, use partial log instead")
                with open(f'outputs_nlu/{dset_subset}_{prompt_id}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        golds.append(row["Gold"])
                print(f"Skipping until {len(preds)}")

            # normalize some labels for more natural prompt:
            if dset_subset == 'imdb_jv_nusantara_text':
                label_names = ['positive', 'negative']
            elif dset_subset == 'indonli_nusantara_pairs':
                label_names = ['no', 'yes', 'maybe']

            en_id_label_map = {
                '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',	'special': 'khusus', 'general': 'umum',
                'no': 'tidak', 'yes': 'ya', 'maybe': 'mungkin', 'negative': 'negatif', 'positive': 'positif', 
                'east': 'timur', 'standard': 'standar', 'ngapak': 'ngapak', 'unknown': 'unknown',
                'neutral': 'netral', 'love': 'cinta', 'fear': 'takut', 'happy': 'senang', 'sad': 'sedih',
                'sadness': 'sedih', 'disgust': 'jijik', 'anger': 'marah', 'surprise': 'terkejut', 'joy': 'senang',
                'reject': 'ditolak', 'tax': 'pajak', 'partial': 'sebagian', 'others': 'lain-lain',
                'granted': 'dikabulkan', 'fulfill': 'penuh', 'correction': 'koreksi',
                'not abusive': 'tidak abusive', 'abusive': 'abusive', 'abusive and offensive': 'abusive dan offensive',
                'support': 'mendukung', 'against': 'bertentangan', 
            }

            if prompt_lang == 'ind':
                label_names = list(map(lambda lab: en_id_label_map[lab], label_names))

            # sample prompt
            print("LABEL NAME = ")
            print(label_names)
            print("SAMPLE PROMPT = ")
            print(to_prompt(tset_dset[0], prompt_template, label_names, prompt_lang))
            print("\n")

            # zero-shot inference
            prompts, labels = [], []
            count = 0
            with torch.inference_mode():
                for sample in tqdm(test_dset):
                    if e < len(preds):
                        continue

                    # Add to buffer
                    prompt_text = to_prompt(sample, prompt_template, label_names, prompt_lang)
                    prompts.append(prompt_text)
                    labels.append(label_to_id_dict[sample['label']] if type(sample['label']) == str else sample['label'])

                    # Batch Inference
                    if len(prompts) == BATCH_SIZE:
                        out = predict_classification(model, tokenizer, prompts, label_names)
                        hyps = argmax(stack(out, axis=-1), axis=-1).tolist()
                        for (prompt_text, hyp, label) in zip(prompts, hyps, labels):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                        prompts, labels = [], []
                        count += 1
                        
                    if count == SAVE_EVERY:
                        # partial saving
                        inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
                        inference_df.to_csv(f'outputs_nlu/{dset_subset}_{prompt_id}_{MODEL.split("/")[-1]}.csv', index=False)
                        count = 0
                        
                if len(prompts) > 0:
                    out = predict_classification(model, tokenizer, prompts, label_names)
                    hyps = argmax(stack(out, axis=-1), axis=-1).tolist()
                    for (prompt_text, hyp, label) in zip(prompts, hyps, labels):
                        inputs.append(prompt_text)
                        preds.append(hyp)
                        golds.append(label)
                    prompts, labels = [], []

            # partial saving
            inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
            inference_df.to_csv(f'outputs_nlu/{dset_subset}_{prompt_id}_{MODEL.split("/")[-1]}.csv', index=False)

        cls_report = classification_report(golds, preds, output_dict=True)
        micro_f1, micro_prec, micro_rec, _ = precision_recall_fscore_support(golds, preds, average='micro')
        print(dset_subset)
        print('accuracy', cls_report['accuracy'])
        print('f1 micro', micro_f1)
        print('f1 macro', cls_report['macro avg']['f1-score'])
        print('f1 weighted', cls_report['weighted avg']['f1-score'])
        print("===\n\n")       
        
        metrics[dset_subset] = {
            'accuracy': cls_report['accuracy'], 
            'micro_prec': micro_prec,
            'micro_rec': micro_rec,
            'micro_f1_score': micro_f1,
            'macro_prec': cls_report['macro avg']['precision'],
            'macro_rec': cls_report['macro avg']['recall'],
            'macro_f1_score': cls_report['macro avg']['f1-score'],
            'weighted_prec': cls_report['weighted avg']['precision'],
            'weighted_rec': cls_report['weighted avg']['recall'],
            'weighted_f1_score': cls_report['weighted avg']['f1-score'],
        }

    pd.DataFrame.from_dict(metrics).T.reset_index().to_csv(f'metrics/nlu_results_{prompt_lang}_{MODEL.split("/")[-1]}.csv', index=False)