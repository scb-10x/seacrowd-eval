#!/bin/bash
echo Eval on $MODEL_NAME
python evaluation/main_nlu_prompt_batch.py tha $MODEL_NAME 8
python evaluation/main_nlg_prompt_batch.py tha $MODEL_NAME 0 8
python evaluation/main_llm_judge_batch.py $MODEL_NAME --data ThaiLLM-Leaderboard/mt-bench-thai
python scripts/transform_result.py $MODEL_NAME
# clear model cache
rm -rf ~/.cache/huggingface/hub/