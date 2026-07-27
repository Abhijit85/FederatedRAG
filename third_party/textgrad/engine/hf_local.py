import logging
import os

from .base import CachedEngine, EngineLM

logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_name: str):
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), "auto")


class ChatHuggingFaceLocal(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = os.getenv("TEXTGRAD_CACHE_DIR")
        if not root:
            root = os.path.join(os.getcwd(), ".cache", "textgrad")
        os.makedirs(root, exist_ok=True)
        cache_path = os.path.join(root, f"cache_hf_local_{model_string.replace('/', '_')}.db")

        super().__init__(cache_path=cache_path)

        dtype_name = os.getenv("TEXTGRAD_HF_DTYPE", "bfloat16")
        local_files_only = os.getenv("TEXTGRAD_HF_LOCAL_FILES_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
        trust_remote_code = os.getenv("TEXTGRAD_HF_TRUST_REMOTE_CODE", "0").strip().lower() in {"1", "true", "yes", "on"}
        attn_impl = os.getenv("TEXTGRAD_HF_ATTENTION", "").strip() or None

        model_kwargs = {
            "torch_dtype": _resolve_dtype(dtype_name),
            "device_map": os.getenv("TEXTGRAD_HF_DEVICE_MAP", "auto"),
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_string,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_string, **model_kwargs)
        self.model.eval()
        self.model_string = model_string
        self.system_prompt = system_prompt

    def generate(self, content, system_prompt=None, **kwargs):
        import torch

        if not isinstance(content, str):
            raise ValueError("ChatHuggingFaceLocal currently supports text-only string prompts.")

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_key = sys_prompt_arg + content
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 512)
        top_p = kwargs.get("top_p", 0.99)

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        model_inputs = {key: value.to(self.model.device) for key, value in model_inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**model_inputs, **generate_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        completion_ids = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
