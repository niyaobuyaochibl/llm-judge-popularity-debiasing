"""LLM judge pipeline: local models and API backends."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "float16"
    max_new_tokens: int = 48
    temperature: float = 0.01
    do_sample: bool = False
    load_in_4bit: bool = False
    use_api: bool = False
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_model: str = "qwen-plus"


@dataclass
class JudgeResult:
    verdict: str
    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    metadata: Dict = field(default_factory=dict)


class DashScopeJudge:
    """Judge backed by DashScope native SDK (for Qwen models)."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    def judge(self, system_prompt: str, user_prompt: str, metadata: Optional[Dict] = None) -> JudgeResult:
        import dashscope

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.perf_counter()
        response = dashscope.Generation.call(
            api_key=self.api_key,
            model=self.config.api_model,
            messages=messages,
            result_format="message",
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            raise RuntimeError(f"DashScope API error {response.code}: {response.message}")

        choice = response.output.choices[0]
        raw_output = (choice.message.content or "").strip()
        usage = response.usage
        verdict = parse_verdict(raw_output)
        return JudgeResult(
            verdict=verdict,
            raw_output=raw_output,
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def cleanup(self) -> None:
        pass


class OpenAICompatibleJudge:
    """Judge backed by any OpenAI-compatible API (Kimi, DeepSeek, MiniMax, etc.)."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        from openai import OpenAI

        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
        self.client = OpenAI(api_key=api_key, base_url=config.api_base, timeout=120)
        self._last_call = 0.0
        model_name = (config.api_model or "").lower()
        api_base = (config.api_base or "").lower()
        is_dashscope = "dashscope" in api_base
        if ("kimi" in model_name or "moonshot" in model_name) and not is_dashscope:
            self._min_interval = 10.0  # ~6 RPM for Moonshot's strict limit
        else:
            self._min_interval = 3.5  # ~17 RPM for DashScope and others

    def judge(self, system_prompt: str, user_prompt: str, metadata: Optional[Dict] = None) -> JudgeResult:
        elapsed_since_last = time.perf_counter() - self._last_call
        if elapsed_since_last < self._min_interval:
            time.sleep(self._min_interval - elapsed_since_last)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        max_retries = 20
        for attempt in range(max_retries):
            try:
                start = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.config.api_model,
                    messages=messages,
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
                self._last_call = time.perf_counter()
                latency_ms = (self._last_call - start) * 1000
                if not response.choices:
                    raw_output = ""
                    verdict = "TIE"
                    return JudgeResult(verdict=verdict, raw_output=raw_output,
                                      prompt_tokens=0, completion_tokens=0,
                                      latency_ms=latency_ms, metadata=metadata or {})
                choice = response.choices[0]
                raw_output = (choice.message.content or "").strip()
                usage = response.usage
                verdict = parse_verdict(raw_output)
                return JudgeResult(
                    verdict=verdict,
                    raw_output=raw_output,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency_ms,
                    metadata=metadata or {},
                )
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    wait = min(self._min_interval * (attempt + 2), 120)
                    print(f"[rate-limit] waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})", flush=True)
                    time.sleep(wait)
                    continue
                if any(code in err_str for code in ("500", "502", "503", "529")) or "timeout" in err_str.lower() or "overload" in err_str.lower():
                    wait = min(10 * (attempt + 1), 180)
                    print(f"[server-error] waiting {wait}s (attempt {attempt+1}/{max_retries}): {err_str[:120]}", flush=True)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"API call failed after {max_retries} retries")

    def cleanup(self) -> None:
        pass


# Backward-compatible alias
APIJudge = DashScopeJudge

DASHSCOPE_MODELS = {"qwen-plus", "qwen-max", "qwen-turbo", "qwen-long",
                    "qwen-plus-latest", "qwen-turbo-latest", "qwen-max-latest"}


def create_judge(config: JudgeConfig):
    """Factory: return the appropriate judge based on config."""
    if config.use_api:
        if config.api_model in DASHSCOPE_MODELS or config.api_model.startswith("qwen"):
            return DashScopeJudge(config)
        return OpenAICompatibleJudge(config)
    return LLMJudge(config)


class LLMJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        is_local_path = Path(self.config.model_name).exists()
        kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if is_local_path:
            kwargs["local_files_only"] = True
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            kwargs["offload_folder"] = "/root/new_paper/results/model_offload"
        tokenizer_kwargs = {"trust_remote_code": True}
        if is_local_path:
            tokenizer_kwargs["local_files_only"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        self.model.eval()

    def judge(self, system_prompt: str, user_prompt: str, metadata: Optional[Dict] = None) -> JudgeResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else None,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        new_tokens = outputs[0][prompt_tokens:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = int(len(new_tokens))
        verdict = parse_verdict(raw_output)
        return JudgeResult(
            verdict=verdict,
            raw_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_verdict(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not text:
        text = raw
    raw_upper = text.strip().upper()
    if raw_upper in {"A", "B", "TIE"}:
        return raw_upper
    first_token = re.match(r"^\s*(A|B|TIE)\b", raw_upper)
    if first_token:
        return first_token.group(1)
    first_line = raw_upper.splitlines()[0].strip() if raw_upper else ""
    if first_line in {"A", "B", "TIE"}:
        return first_line
    match = re.search(r"VERDICT:\s*(A|B|TIE)", raw_upper)
    if match:
        return match.group(1)
    if "LIST A" in raw_upper and "LIST B" not in raw_upper:
        return "A"
    if "LIST B" in raw_upper and "LIST A" not in raw_upper:
        return "B"
    for ch in raw_upper:
        if ch in {"A", "B"}:
            return ch
    return "UNPARSEABLE"
