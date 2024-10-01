import os, sys
import csv
import string

from os.path import exists
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

import torch

from transformers import set_seed
from model_utils import load_model_runner
from prompt_utils import get_prompt, get_label_mapping
from data_utils import load_nlu_datasets

csv.field_size_limit(sys.maxsize)


def to_prompt(input, prompt, labels, prompt_lang, schema):
    if schema == "text" or schema == "pairs":
        # single label
        if "text" in input:
            prompt = prompt.replace("[INPUT]", input["text"].strip())
        else:
            prompt = prompt.replace("[INPUT_A]", input["text_1"].strip())
            prompt = prompt.replace("[INPUT_B]", input["text_2"].strip())

        # replace [OPTIONS] to A, B, C
        if "[OPTIONS]" in prompt:
            new_labels = [f"{l}" for l in labels]
            if len(new_labels) > 2:
                prompt = prompt.replace("[OPTIONS]", ", ".join(new_labels))
            else:
                prompt = prompt.replace("[OPTIONS]", " ".join(new_labels))
    elif schema == "qa":
        if "[CONTEXT]" in prompt:
            context = (
                ""
                if "context" not in input.keys() or input["context"] is None
                else input["context"].strip()
            )
            prompt = prompt.replace("[CONTEXT]", context)
        prompt = prompt.replace("[QUESTION]", input["question"].strip())

        choices = ""
        for i, choice in enumerate(input["choices"]):
            if i > 0:
                choices += "\n"
            choices += f"{string.ascii_lowercase[i]}. {choice.strip()}"
        prompt = prompt.replace("[ANSWER_CHOICES]", choices)
    else:
        raise ValueError("Only support `text`, `pairs`, and `qa` schemas.")

    return prompt


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError(
            "main_nlu_prompt.py <prompt_lang> <model_path_or_name> <batch_size>"
        )
    if len(sys.argv) > 6:
        raise ValueError(
            "main_nlu_prompt.py <prompt_lang> <model_path_or_name> <batch_size> <base_url> <api_key>"
        )

    prompt_lang = sys.argv[1]
    MODEL = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])
    BASE_URL = None
    API_KEY = None
    OPENAI_COMPATIBLE = False
    if len(sys.argv) == 6:
        BASE_URL = sys.argv[4]
        API_KEY = sys.argv[5]
        OPENAI_COMPATIBLE = True

    out_dir = "./outputs_nlu"
    metric_dir = "./metrics_nlu"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # Load Prompt
    TASK_TYPE_TO_PROMPT = get_prompt(prompt_lang)

    # Load Dataset
    print("Load NLU Datasets...")
    nlu_datasets = load_nlu_datasets()

    print(f"Loaded {len(nlu_datasets)} NLU datasets")
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f"{i} {dset_subset}")

    # Set seed before initializing model.
    set_seed(42)

    # model_runner = load_model_runner(MODEL)
    model_runner = load_model_runner(
        MODEL,
        openai_compatible=OPENAI_COMPATIBLE,
        base_url=BASE_URL,
        api_key=API_KEY,
        fast=True,
    )

    with torch.no_grad():
        metrics = []
        labels = []
        for i, dset_subset in enumerate(nlu_datasets.keys()):
            print(f"=====({i+1}/{len(nlu_datasets.keys())}) {dset_subset} =====")
            schema = dset_subset.split("_")[-1]
            nlu_dset, task_type = nlu_datasets[dset_subset]
            if task_type.value not in TASK_TYPE_TO_PROMPT:
                print(f"SKIPPING {dset_subset}")
                continue

            # Retrieve metadata
            split = "test"
            if "test" in nlu_dset.keys():
                test_dset = nlu_dset["test"]
            else:
                test_dset = nlu_dset["train"]
                split = "train"
            print(f"Processing {dset_subset}")

            # Add `label` based on `answer` for QA
            if schema == "qa":
                correct_answer_indices = []
                exclude_idx = []
                for i in range(len(test_dset)):
                    if isinstance(test_dset[i]["answer"], list):
                        try:
                            correct_answer_indices += [
                                test_dset[i]["choices"].index(test_dset[i]["answer"][0])
                            ]
                        except:
                            exclude_idx.append(i)
                    else:
                        correct_answer_indices += [
                            test_dset[i]["choices"].index(test_dset[i]["answer"])
                        ]
                test_dset = test_dset.select(
                    (i for i in range(len(test_dset)) if i not in set(exclude_idx))
                )
                test_dset = test_dset.add_column("label", correct_answer_indices)

            # Retrieve & preprocess labels
            try:
                label_names = test_dset.features["label"].names
            except:
                label_names = list(set(test_dset["label"]))

            # normalize some labels for more natural prompt
            label_mapping = get_label_mapping(dset_subset, prompt_lang)
            label_names = list(map(lambda x: label_mapping[x], label_mapping))

            label_to_id_dict = {l: i for i, l in enumerate(label_names)}

            for prompt_id, prompt_template in enumerate(
                TASK_TYPE_TO_PROMPT[task_type.value]
            ):
                inputs, preds, golds = [], [], []

                # Check saved data
                if exists(
                    f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{MODEL.split("/")[-1]}.csv'
                ):
                    print("Output exist, use partial log instead")
                    with open(
                        f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{MODEL.split("/")[-1]}.csv'
                    ) as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            inputs.append(row["Input"])
                            preds.append(row["Pred"])
                            golds.append(row["Gold"])
                    print(f"Skipping until {len(preds)}")

                # sample prompt
                print("= LABEL NAME =")
                print(label_names)
                print("= SAMPLE PROMPT =")

                print(
                    to_prompt(
                        test_dset[0], prompt_template, label_names, prompt_lang, schema
                    )
                )
                print("\n")

                # zero-shot inference
                prompts, labels = [], []
                count = 0
                with torch.inference_mode():
                    for e, sample in tqdm(enumerate(test_dset), total=len(test_dset)):
                        if e < len(preds):
                            continue

                        prompt_text = to_prompt(
                            sample, prompt_template, label_names, prompt_lang, schema
                        )
                        prompts.append(prompt_text)
                        labels.append(
                            label_to_id_dict[sample["label"]]
                            if type(sample["label"]) == str
                            else sample["label"]
                        )

                        # Batch Inference
                        if len(prompts) == BATCH_SIZE:
                            hyps = model_runner.predict_classification(
                                prompts, label_names, BATCH_SIZE=BATCH_SIZE
                            )
                            for prompt_text, hyp, label in zip(prompts, hyps, labels):
                                inputs.append(prompt_text)
                                preds.append(hyp)
                                golds.append(label)
                            prompts, labels = [], []
                            count += 1

                    if len(prompts) > 0:
                        hyps = model_runner.predict_classification(
                            prompts, label_names, BATCH_SIZE=BATCH_SIZE
                        )
                        for prompt_text, hyp, label in zip(prompts, hyps, labels):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                        prompts, labels = [], []

                # partial saving
                inference_df = pd.DataFrame(
                    list(zip(inputs, preds, golds)), columns=["Input", "Pred", "Gold"]
                )
                inference_df.to_csv(
                    f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{MODEL.split("/")[-1]}.csv',
                    index=False,
                )

                cls_report = classification_report(golds, preds, output_dict=True)
                micro_f1, micro_prec, micro_rec, _ = precision_recall_fscore_support(
                    golds, preds, average="micro"
                )
                print(dset_subset)
                print("accuracy", cls_report["accuracy"])
                print("f1 micro", micro_f1)
                print("f1 macro", cls_report["macro avg"]["f1-score"])
                print("f1 weighted", cls_report["weighted avg"]["f1-score"])
                print("===\n\n")

                metrics.append(
                    {
                        "dataset": dset_subset,
                        "prompt_id": prompt_id,
                        "prompt_lang": prompt_lang,
                        "accuracy": cls_report["accuracy"],
                        "micro_prec": micro_prec,
                        "micro_rec": micro_rec,
                        "micro_f1_score": micro_f1,
                        "macro_prec": cls_report["macro avg"]["precision"],
                        "macro_rec": cls_report["macro avg"]["recall"],
                        "macro_f1_score": cls_report["macro avg"]["f1-score"],
                        "weighted_prec": cls_report["weighted avg"]["precision"],
                        "weighted_rec": cls_report["weighted avg"]["recall"],
                        "weighted_f1_score": cls_report["weighted avg"]["f1-score"],
                    }
                )

    pd.DataFrame(metrics).reset_index().to_csv(
        f'{metric_dir}/nlu_results_{prompt_lang}_{MODEL.split("/")[-1]}.csv',
        index=False,
    )
