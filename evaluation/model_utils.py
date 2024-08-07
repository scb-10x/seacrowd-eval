from typing import Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import torch

def load_model_and_tokenizer(model_name_or_path: str, compile=False) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    
    if compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            pass

    model.eval()
    return model, tokenizer
    