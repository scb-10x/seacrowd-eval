
# ThaiLLM-Leaderboard Eval Runner

The Thai-LLM Leaderboard 🇹🇭 focuses on standardizing evaluation methods for large language models (LLMs) in the Thai language based on [Seacrowd](https://github.com/SEACrowd/seacrowd-experiments). As part of an open community project, we welcome you to submit new evaluation tasks or models.

## Run an Eval

### Install
```sh
pip install -r requirements.txt
```

### Run Eval
```sh
MODEL_NAME=airesearch/LLaMa3-8b-WangchanX-sft-Full sh runner.sh
```

### Submit Eval Result
```sh
python scripts/transform_result.py $MODEL_NAME
```

## Develop an Eval

### New Dataset Based on the Same Pipeline (NLU, NLG, LLM as Judge)

1. Edit `evaluation/data_utils.py` to include your evaluation dataset.
2. Create a pull request on [https://huggingface.co/spaces/ThaiLLM-Leaderboard/leaderboard](https://huggingface.co/spaces/ThaiLLM-Leaderboard/leaderboard) by adding a result key to human-readable name mapper in `leaderboard/read_evals.py`.

### New Eval Pipeline

1. Create a file similar to `evaluation/main_**_batch.py` to run an evaluation and output its results.
2. Add a method in `scripts/transform_result.py` to transform the evaluation result into the same format as the example below.

#### Example Output File After Transform

```json
{
  "config": {
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"
  },
  "results": {
    "xcopa_tha_seacrowd_qa": {
      "accuracy": 0.522
    },
    "wisesight_thai_sentiment_seacrowd_text": {
      "accuracy": 0.4545114189442156
    },
    "belebele_tha_thai_seacrowd_qa": {
      "accuracy": 0.4177777777777778
    },
    "xnli.tha_seacrowd_pairs": {
      "accuracy": 0.3407185628742515
    }
  }
}
```

3. Create a pull request on [https://huggingface.co/spaces/ThaiLLM-Leaderboard/leaderboard](https://huggingface.co/spaces/ThaiLLM-Leaderboard/leaderboard) by adding a result key to human-readable name mapper in `leaderboard/read_evals.py`.