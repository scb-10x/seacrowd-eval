#!/bin/bash
echo Eval on $MODEL_NAME
python evaluation/main_nlu_prompt_batch.py tha $MODEL_NAME 4
python evaluation/main_nlg_prompt_batch.py tha $MODEL_NAME 0 4
python evaluation/main_llm_judge_batch.py $MODEL_NAME --data ThaiLLM-Leaderboard/mt-bench-thai
python scripts/transform_result.py $MODEL_NAME
# clear model cache
rm -rf ~/.cache/huggingface/hub/

# run with OpenAI API compatible
# echo Eval on $MODEL_NAME
# python evaluation/main_nlu_prompt_batch.py tha $MODEL_NAME 4 https://api.xx.xx/v1 apikey-xxxxx
# python evaluation/main_nlg_prompt_batch.py tha $MODEL_NAME 0 4 https://api.xx.xx/v1 apikey-xxxxx
# python evaluation/main_llm_judge_batch.py $MODEL_NAME --data ThaiLLM-Leaderboard/mt-bench-thai --base_url https://api.xx.xx/v1 --api_key apikey-xxxxx
# python scripts/transform_result.py $MODEL_NAME
# # clear model cache
# rm -rf ~/.cache/huggingface/hub/
