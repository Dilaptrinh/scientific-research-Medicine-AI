"""Tải mô hình ngôn ngữ (LLM) và sinh câu trả lời."""
from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(self, llm_model: str, load_in_4bit: bool = False) -> None:
        print(f"Tải LLM {llm_model} (lần đầu có thể mất vài phút) ...")
        self.tok = AutoTokenizer.from_pretrained(llm_model)
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig   # cần: pip install bitsandbytes
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model, **kwargs)
        print("Đã tải LLM:", llm_model)

    def sinh(self, messages: List[Dict[str, str]], max_tokens: int = 450) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            out = self.llm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                    repetition_penalty=1.05)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
