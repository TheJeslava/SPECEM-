from __future__ import annotations

import gc
import inspect
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


@dataclass
class GenerationResult:
    text: str
    elapsed_s: float
    new_tokens: int
    peak_vram_gb: float


class UnifiedTextGenerator:
    def __init__(self, model_id: str, hf_id: str, quantized: bool = True):
        self.model_id = model_id
        self.hf_id = hf_id
        self.quantized = quantized
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.mode = "causal_lm"
        self.family = self._detect_family()

    def _detect_family(self) -> str:
        lowered = self.hf_id.lower()
        if "qwen" in lowered:
            return "qwen"
        if "gemma" in lowered:
            return "gemma"
        if "llama" in lowered:
            return "llama"
        if "mimo" in lowered:
            return "mimo"
        if "mistral" in lowered or "ministral" in lowered:
            return "mistral"
        return "generic"

    def load(self) -> None:
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if "mistral" in self.hf_id.lower() or "ministral" in self.hf_id.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True
        if self.quantized:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, **tokenizer_kwargs)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kwargs).eval()
            self.mode = "causal_lm"
            return
        except Exception as causal_exc:
            try:
                self.processor = AutoProcessor.from_pretrained(self.hf_id, **tokenizer_kwargs)
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(self.hf_id, **kwargs).eval()
            self.mode = "image_text_to_text"
            self._causal_error = repr(causal_exc)

    def unload(self) -> None:
        del self.model
        self.model = None
        self.tokenizer = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_inputs(self, instruction: str) -> dict[str, torch.Tensor]:
        if self.family == "mimo" and self.mode == "causal_lm":
            text = self._plain_mimo_prompt(instruction)
            encoded = self.tokenizer(text, return_tensors="pt")
            return self._normalize_inputs(encoded)

        messages = self._messages(instruction)
        if self.mode == "causal_lm":
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    text = self._apply_chat_template(self.tokenizer, messages)
                    encoded = self.tokenizer(text, return_tensors="pt")
                except TypeError:
                    encoded = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
            else:
                encoded = self.tokenizer(instruction, return_tensors="pt")
            return self._normalize_inputs(encoded)

        if hasattr(self.processor, "apply_chat_template"):
            try:
                text = self._apply_chat_template(self.processor, messages)
            except TypeError:
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            text = instruction
        return self._normalize_inputs(self.processor(text=text, return_tensors="pt"))

    def _messages(self, instruction: str) -> list[dict[str, str]]:
        system = (
            "You are a helpful assistant. Answer the user's task directly. "
            "Only output the final answer. Do not output analysis, reasoning steps, "
            "thinking process, hidden thoughts, or meta commentary."
        )
        if self.family == "gemma":
            return [{"role": "user", "content": f"{system}\n\n{instruction}"}]
        if self.family == "mimo":
            mimo_instruction = (
                "Answer directly and concisely. Do not use <think> tags. "
                "Do not write reasoning, analysis, hidden thoughts, or planning. "
                "Return only the final answer.\n\n"
                f"{instruction}"
            )
            return [{"role": "system", "content": ""}, {"role": "user", "content": mimo_instruction}]
        return [{"role": "system", "content": system}, {"role": "user", "content": instruction}]

    def _plain_mimo_prompt(self, instruction: str) -> str:
        return (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n"
            "Answer the following task directly and concisely. Do not include reasoning, "
            "analysis, planning, self-evaluation, repetition, or think tags.\n\n"
            f"{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def _apply_chat_template(self, template_owner: Any, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        try:
            params = inspect.signature(template_owner.apply_chat_template).parameters
        except (TypeError, ValueError):
            params = {}
        if "enable_thinking" in params:
            kwargs["enable_thinking"] = False
        elif self.family == "qwen":
            try:
                return template_owner.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                pass
        return template_owner.apply_chat_template(messages, **kwargs)

    def _normalize_inputs(self, inputs) -> dict[str, torch.Tensor]:
        if torch.is_tensor(inputs):
            return {"input_ids": inputs.to(self.model.device)}
        if hasattr(inputs, "items"):
            normalized = {}
            for key, value in inputs.items():
                if hasattr(value, "to") and hasattr(value, "shape"):
                    normalized[key] = value.to(self.model.device)
            if "input_ids" not in normalized and hasattr(inputs, "input_ids"):
                normalized["input_ids"] = inputs.input_ids.to(self.model.device)
            return normalized
        if hasattr(inputs, "input_ids"):
            normalized = {"input_ids": inputs.input_ids.to(self.model.device)}
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                normalized["attention_mask"] = inputs.attention_mask.to(self.model.device)
            return normalized
        normalized = {}
        for key, value in dict(inputs).items():
            if hasattr(value, "to") and hasattr(value, "shape"):
                normalized[key] = value.to(self.model.device)
            elif hasattr(value, "input_ids"):
                normalized[key] = value.input_ids.to(self.model.device)
        if "input_ids" not in normalized and hasattr(inputs, "input_ids"):
            normalized["input_ids"] = inputs.input_ids.to(self.model.device)
        return normalized

    def generate(
        self,
        instruction: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool = False,
    ) -> GenerationResult:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inputs = self._build_inputs(instruction)
        input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        tokenizer = self.tokenizer if self.tokenizer is not None else getattr(self.processor, "tokenizer", None)
        eos_token_id = self._generation_eos_token_id(tokenizer)
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(self.model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        start = time.time()
        with torch.inference_mode():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": pad_token_id,
            }
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            if do_sample:
                generation_kwargs.update({"temperature": temperature, "top_p": top_p})
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        elapsed = time.time() - start
        generated = output[0][input_len:] if input_len else output[0]
        text = clean_generated_text(tokenizer.decode(generated, skip_special_tokens=True).strip())
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        return GenerationResult(text=text, elapsed_s=elapsed, new_tokens=int(generated.shape[-1]), peak_vram_gb=peak)

    def _generation_eos_token_id(self, tokenizer: Any) -> int | list[int] | None:
        candidates = [
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
            getattr(self.model.config, "eos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, tuple):
                candidate = list(candidate)
            if isinstance(candidate, list):
                values = [int(value) for value in candidate if value is not None]
                if values:
                    return values
            return int(candidate)
        return None


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows if limit is None else rows[:limit]


def model_cache_dir(hf_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    return hf_home / "hub" / ("models--" + hf_id.replace("/", "--"))


def remove_model_cache(hf_id: str) -> None:
    path = model_cache_dir(hf_id)
    if path.exists():
        shutil.rmtree(path)


def clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^<think>.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    markers = [
        "Final Answer:",
        "Final answer:",
        "Answer:",
        "答案：",
        "最终答案：",
    ]
    lowered = cleaned.lower()
    if (
        "thinking process" in lowered
        or "analyze the request" in lowered
        or cleaned.startswith("Thinking Process:")
    ):
        for marker in markers:
            pos = cleaned.rfind(marker)
            if pos != -1:
                return cleaned[pos + len(marker):].strip()
    return cleaned
