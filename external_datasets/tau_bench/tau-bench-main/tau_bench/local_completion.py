from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch
from litellm import completion as litellm_completion
from transformers import AutoModelForCausalLM, AutoTokenizer


LOCAL_PROVIDER = "local-hf"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

MODEL_ALIASES = {
    "meta-llama/llama-3.1-8b-instruct": os.environ.get(
        "LOCAL_LLAMA31_MODEL_PATH",
        "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "qwen/qwen2.5-7b-instruct": "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "qwen/qwen2.5-14b-instruct": "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-14B-Instruct",
    "meta-llama/llama-3.3-70b-instruct": "/mnt/shared/shared_hf_home/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
}

ROLE_STOP_MARKERS = [
    "\nUSER:",
    "\nASSISTANT:",
    "\nSYSTEM:",
    "\nTOOL:",
    "\nHuman:",
    "\nCustomer:",
    "\nAgent:",
    "<|im_end|>",
]


@dataclass
class _Message:
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def model_dump(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        return payload


@dataclass
class _Choice:
    message: _Message


class _Response:
    def __init__(self, message: _Message) -> None:
        self.choices = [_Choice(message=message)]
        self._hidden_params = {"response_cost": 0.0}


def _resolve_model_path(model: Optional[str]) -> str:
    model_name = (model or DEFAULT_LOCAL_MODEL).strip()
    override = os.environ.get("LOCAL_HF_MODEL_PATH")
    if override:
        return override
    alias = MODEL_ALIASES.get(model_name.lower())
    if alias:
        return alias
    return model_name


@lru_cache(maxsize=4)
def _load_local_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def _tool_prompt(tools: List[Dict[str, Any]]) -> str:
    rendered = []
    for tool in tools:
        fn = tool.get("function") or {}
        rendered.append(
            json.dumps(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                },
                ensure_ascii=True,
            )
        )
    return (
        "You may either respond to the user or call exactly one tool.\n"
        "If you call a tool, output exactly one compact JSON object and nothing else in this form:\n"
        '{"tool_call":{"name":"tool_name","arguments":{"key":"value"}}}\n'
        "If you reply to the user, output exactly one compact JSON object and nothing else in this form:\n"
        '{"response":"plain text to send to the user"}\n'
        "Never emit multi-turn transcripts, role labels, or explanatory preambles.\n"
        "Available tools:\n" + "\n".join(rendered)
    )


def _normalize_messages(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if content is None and message.get("tool_calls"):
            content = json.dumps({"tool_calls": message["tool_calls"]}, ensure_ascii=True)
        normalized.append({"role": role, "content": "" if content is None else str(content)})
    if tools:
        normalized.append({"role": "system", "content": _tool_prompt(tools)})
    return normalized


def _render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _trim_generation(text: str) -> str:
    trimmed = text.strip()
    for marker in ROLE_STOP_MARKERS:
        idx = trimmed.find(marker)
        if idx != -1:
            trimmed = trimmed[:idx].strip()
    if "###STOP###" in trimmed and trimmed.strip() != "###STOP###":
        trimmed = trimmed.split("###STOP###", 1)[0].strip()
    return trimmed.strip()


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    candidates = [text]
    if "```json" in text:
        candidates.extend(re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            pass
        else:
            if isinstance(parsed, dict):
                return parsed
        for start in [m.start() for m in re.finditer(r"\{", candidate)]:
            try:
                parsed, _ = decoder.raw_decode(candidate[start:])
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _normalize_tool_calls_payload(parsed: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if isinstance(parsed.get("tool_call"), dict):
        tool_call = parsed["tool_call"]
        return [{
            "id": "local_tool_call_0",
            "type": "function",
            "function": {
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("arguments", {}), ensure_ascii=True),
            },
        }]
    tool_calls = parsed.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        normalized = []
        for idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    json.loads(args)
                except Exception:
                    pass
                else:
                    args = json.loads(args)
            normalized.append({
                "id": tool_call.get("id") or f"local_tool_call_{idx}",
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": fn.get("name") or tool_call.get("name"),
                    "arguments": json.dumps(args, ensure_ascii=True),
                },
            })
        return normalized or None
    return None


def _build_message_from_output(text: str) -> _Message:
    trimmed = _trim_generation(text)
    parsed = _extract_json_block(trimmed)
    if parsed:
        normalized_tool_calls = _normalize_tool_calls_payload(parsed)
        if normalized_tool_calls:
            return _Message(role="assistant", content=None, tool_calls=normalized_tool_calls)
        if "response" in parsed:
            return _Message(role="assistant", content=str(parsed["response"]).strip())
    return _Message(role="assistant", content=trimmed)


def _local_completion(
    model: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    **_: Any,
) -> _Response:
    model_path = _resolve_model_path(model)
    tokenizer, model_obj = _load_local_model(model_path)
    prompt_messages = _normalize_messages(messages or [], tools)
    prompt = _render_prompt(tokenizer, prompt_messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}
    generation_max_tokens = min(max_tokens, 192 if tools else 96)
    with torch.no_grad():
        output_ids = model_obj.generate(
            **inputs,
            max_new_tokens=generation_max_tokens,
            do_sample=temperature > 1e-5,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return _Response(_build_message_from_output(text))


def completion(*args: Any, **kwargs: Any) -> Any:
    provider = kwargs.get("custom_llm_provider")
    if provider == LOCAL_PROVIDER:
        return _local_completion(*args, **kwargs)
    return litellm_completion(*args, **kwargs)
