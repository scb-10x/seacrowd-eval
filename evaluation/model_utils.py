import abc
import copy
from dataclasses import dataclass
import dataclasses
from multiprocessing import Pool
import os
from typing import Dict, Optional, List, Union
import numpy as np
from retry import retry
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import torch.nn.functional as F
import logging
from functools import partial

try:
    import google.generativeai as genai
except ImportError:
    pass

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

MAX_GENERATION_LENGTH = 512

openai_client = None
anthropic_client = None


@retry(Exception, tries=10, delay=20)
def _call_openai(
    messages: List[Dict[str, str]],
    model_name: str,
    max_tokens=200,
    temperature=0.0,
    **kwargs
):
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        seed=42,
        max_tokens=max_tokens,
        **kwargs
    )
    return completion.choices[0].message.content

@retry(Exception, tries=10, delay=20)
def _call_anthropic(
    messages: List[Dict[str, str]],
    model_name: str,
    max_tokens=200,
    temperature=0.0,
    **kwargs
):
    system_kwargs = {}
    if messages[0]['role'] == 'system':
        system_msg = messages.pop(0)
        system_kwargs['system'] = system_msg['content']
    message = anthropic_client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=messages,
        temperature=temperature,
        **system_kwargs,
        **kwargs
    )
    return message.content[0].text

@retry(Exception, tries=10, delay=20)
def _call_gemini(
    messages: List[Dict[str, str]],
    model_name: str,
    max_tokens=200,
    temperature=0.0,
    **kwargs
):
    messages = copy.deepcopy(messages)
    history = []
    if messages[0]["role"] == "system":
        system_msg = messages.pop(0)
        gemini_client = genai.GenerativeModel(
            model_name, system_instruction=system_msg["content"]
        )
    else:
        gemini_client = genai.GenerativeModel(model_name)
        
    safe = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    last_msg = messages.pop(-1)
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": msg["content"]})
    
    chat = gemini_client.start_chat(history=history)
    response = chat.send_message(
        last_msg["content"],
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        ),
        safety_settings=safe
    )
    return response.text


# Copy from vllm project
def _get_and_verify_max_len(
    hf_config,
    max_model_len: Optional[int] = None,
    disable_sliding_window: bool = False,
    sliding_window_len: Optional[int] = None,
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if disable_sliding_window and sliding_window_len is not None:
        max_len_key = (
            "sliding_window"
            if sliding_window_len < derived_max_model_len
            else max_len_key
        )
        derived_max_model_len = min(derived_max_model_len, sliding_window_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        print(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            raise ValueError("rope_scaling must have a 'type' or 'rope_type' key.")

        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate."
                )

            assert "factor" in rope_scaling
            scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling["original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
    max_model_len = min(max_model_len, derived_max_model_len)
    return int(max_model_len)


@dataclass
class ChatMessage:
    role: str
    content: str


class AbsModel(abc.ABC):

    def __init__():
        pass

    def predict_classification(
        self, prompts: List[str], labels: List[str]
    ) -> List[int]:
        raise NotImplementedError()

    def predict_generation(
        self, prompts: List[Union[str, ChatMessage]], **kwargs
    ) -> List[str]:
        raise NotImplementedError()


class APIModel(AbsModel):

    def __init__(self, model_name):
        self.model_name = model_name
        self.generate_fn = None
        if "gpt" in self.model_name:
            global openai_client
            from openai import OpenAI

            openai_client = OpenAI()
            self.generate_fn = _call_openai
        elif "claude" in self.model_name:
            global anthropic_client
            from anthropic import Anthropic

            anthropic_client = Anthropic()
            self.generate_fn = _call_anthropic
        elif "gemini" in self.model_name:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.generate_fn = _call_gemini
        else:
            raise NotImplementedError()
        self.max_generation_length = MAX_GENERATION_LENGTH

    def predict_classification(
        self, prompts: List[str], labels: List[str]
    ):  # return List[len(prompts)] with int value as idx of each
        inputs = [
            [
                {
                    "role": "system",
                    "content": "Answer with only exact answer specified by instruction. The following are multiple choice questions.",
                },
                {"role": "user", "content": prompt.replace("[LABEL_CHOICE]", "")},
            ]
            for prompt in prompts
        ]
        hyps = []
        with Pool(min(8, len(prompts))) as p:
            _generate_fn = partial(self.generate_fn, model_name=self.model_name)
            results = p.map(_generate_fn, inputs)

        for response, _prompt in zip(results, prompts):
            selected_idx = -1
            for i, label_name in enumerate(labels):
                label_lower = label_name.strip().lower()
                response_lower = response.strip().lower()
                if response_lower == label_lower or response_lower.startswith(
                    label_lower
                ):
                    selected_idx = i
                    break
            hyps.append(selected_idx)
        assert len(prompts) == len(hyps)
        return hyps

    def predict_generation(
        self, prompts: List[Union[str, ChatMessage]], **kwargs
    ) -> List[str]:
        if isinstance(prompts[0], str):
            prompts = [
                [
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
        else:
            prompts = [[dataclasses.asdict(p) for p in conv] for conv in prompts]

        with Pool(min(8, len(prompts))) as p:
            _generate_fn = partial(
                self.generate_fn,
                model_name=self.model_name,
                max_tokens=self.max_generation_length,
                **kwargs
            )
            results = p.map(_generate_fn, prompts)
        return results


class HFModel(AbsModel):

    def __init__(self, model_name_or_path: str, compile=False):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        tokenizer.padding_side = "left"
        model_max_length = min(_get_and_verify_max_len(model.config), 8192)
        self.max_generation_length = MAX_GENERATION_LENGTH
        self.model_max_length = model_max_length
        print("model_max_length", self.model_max_length)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.bos_token
                if tokenizer.bos_token is not None
                else tokenizer.eos_token
            )

        if compile:
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                pass

        model.eval()
        self.model_name = model_name_or_path
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def _get_logprobs(
        self, model, model_name, tokenizer, inputs, label_ids=None, label_attn=None
    ):
        inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length - self.max_generation_length,
        ).to("cuda")
        if "sea-lion" in model_name and "token_type_ids" in inputs.keys():
            del inputs["token_type_ids"]
        logits = model(**inputs).logits
        output_ids = inputs["input_ids"][:, 1:]
        logprobs = torch.gather(
            F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)
        ).squeeze(dim=-1)
        logprobs[inputs["attention_mask"][:, :-1] == 0] = 0
        return logprobs.sum(dim=1).cpu()

    @torch.inference_mode()
    def predict_classification(self, prompts: List[str], labels: List[str]):
        probs = []
        for label in labels:
            inputs = [prompt.replace("[LABEL_CHOICE]", label) for prompt in prompts]
            probs.append(
                self._get_logprobs(self.model, self.model_name, self.tokenizer, inputs)
                .float()
                .numpy()
            )
        result = np.argmax(np.stack(probs, axis=-1), axis=-1).tolist()
        return result

    def _get_terminator(self):
        eos_tokens = ["<|eot_id|>", "<|im_start|>", "<|im_end|>"]
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        for t in eos_tokens:
            tok = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tok, int):
                terminators.append(tok)
        return terminators

    @torch.inference_mode()
    def predict_generation(self, prompts: List[Union[str, ChatMessage]], **kwargs):
        if isinstance(prompts[0], str):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in prompts
            ]
        else:
            prompts = [
                self.tokenizer.apply_chat_template(
                    [dataclasses.asdict(p) for p in conv],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for conv in prompts
            ]      

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length - self.max_generation_length,
        ).to(self.model.device)

        input_sizes = inputs["input_ids"].shape[-1]

        if "sea-lion" in self.model_name and "token_type_ids" in inputs.keys():
            del inputs["token_type_ids"]

        temperature = kwargs.pop("temperature", 0.2)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=self.max_generation_length,
            eos_token_id=self._get_terminator(),
            **kwargs
        )
        preds = self.tokenizer.batch_decode(
            outputs[:, input_sizes:], skip_special_tokens=True
        )
        return preds


def load_model_runner(model_name: str, fast=False):
    if model_name in [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "claude-3-5-sonnet-20240620",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro-exp-0827",
        "gemini-1.5-pro-001"
    ]:
        model_runner = APIModel(model_name)
    else:
        model_runner = HFModel(model_name, compile=fast)
    return model_runner
