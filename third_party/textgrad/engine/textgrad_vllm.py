try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "If you'd like to use VLLM models, please install the vllm package by running `pip install vllm` or `pip install textgrad[vllm]."
    )

import json
import os
import shutil
import platformdirs
from .base import EngineLM, CachedEngine

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, constants


class ChatVLLM(EngineLM, CachedEngine):
    # Default system prompt for VLLM models
    DEFAULT_SYSTEM_PROMPT = ""

    def __init__(
        self,
        model_string="meta-llama/Meta-Llama-3.2-11B-Instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ):
        self.model_string = model_string
        self.system_prompt = system_prompt
        preferred_root = os.environ.get("TEXTGRAD_CACHE_DIR") or platformdirs.user_cache_dir("textgrad")
        root = preferred_root
        try:
            os.makedirs(root, exist_ok=True)
        except OSError:
            root = os.path.join(os.getcwd(), ".cache", "textgrad")
            os.makedirs(root, exist_ok=True)
        safe_model_id = model_string.replace("/", "__")
        cache_path = os.path.join(root, safe_model_id)
        try:
            os.makedirs(cache_path, exist_ok=True)
        except OSError:
            root = os.path.join(os.getcwd(), ".cache", "textgrad")
            os.makedirs(root, exist_ok=True)
            cache_path = os.path.join(root, safe_model_id)
            os.makedirs(cache_path, exist_ok=True)
        super().__init__(cache_path=cache_path)

        self.model_cache_root = cache_path
        self.model_artifact_dir = os.path.join(self.model_cache_root, "hf")
        os.makedirs(self.model_artifact_dir, exist_ok=True)
        self.hf_cache_dir = os.path.join(self.model_cache_root, "hf_cache")
        os.makedirs(self.hf_cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", self.hf_cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.hf_cache_dir)

        self.rope_scaling_override = self._resolve_rope_scaling()
        self._ensure_config_compatibility()

        # self._check_and_download_model(self.model_string)

        # Instantiate vLLM with the requested model and a dedicated download cache
        llm_kwargs = {}
        if self.rope_scaling_override:
            llm_kwargs["rope_scaling"] = self.rope_scaling_override

        self.client = LLM(model=self.model_string, download_dir=self.model_artifact_dir, **llm_kwargs)
        self.tokenizer = self.client.get_tokenizer()

    def _check_and_download_model(self, model_string):
        """
        Check if the model exists locally, and if not, download it.
        """
        model_cache_dir = platformdirs.user_cache_dir("huggingface", "models")
        model_path = os.path.join(model_cache_dir, model_string)
        
        # If the model directory doesn't exist, download it.
        if not os.path.exists(model_path):
            print(f"Model '{model_string}' not found locally. Downloading...")
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # Download the model and tokenizer using Hugging Face API
            try:
                AutoModelForCausalLM.from_pretrained(model_string, cache_dir=model_cache_dir)
                AutoTokenizer.from_pretrained(model_string, cache_dir=model_cache_dir)
                print(f"Model '{model_string}' downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model '{model_string}': {e}")
        else:
            print(f"Model '{model_string}' found locally.")

    def _resolve_rope_scaling(self):
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                self.model_string,
                cache_dir=self.hf_cache_dir,
                token=token,
            )
        except Exception:
            config = None

        if config is not None:
            rope_scaling = getattr(config, "rope_scaling", None)
            rope = self._normalize_rope_scaling(rope_scaling)
            if rope:
                return rope

        cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE", constants.HUGGINGFACE_HUB_CACHE)
        repo_dir = self.model_string.replace("/", "--")
        snapshot_dir = os.path.join(cache_root, f"models--{repo_dir}", "snapshots")
        if os.path.isdir(snapshot_dir):
            for entry in sorted(os.listdir(snapshot_dir), reverse=True):
                cfg_path = os.path.join(snapshot_dir, entry, "config.json")
                rope = self._load_rope_scaling_from_file(cfg_path)
                if rope:
                    return rope

        local_cfg = os.path.join(self.model_artifact_dir, "config.json")
        return self._load_rope_scaling_from_file(local_cfg)

    def _ensure_config_compatibility(self) -> None:
        """
        Ensure the downloaded HF config matches what vLLM expects.
        """
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        snapshot_path = None
        try:
            snapshot_path = snapshot_download(
                repo_id=self.model_string,
                cache_dir=self.hf_cache_dir,
                local_dir=self.model_artifact_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["config.json"],
                token=token,
            )
        except Exception:
            # If download fails (offline, already cached), continue gracefully.
            pass

        candidates = []
        if snapshot_path:
            cfg = os.path.join(snapshot_path, "config.json")
            if os.path.exists(cfg):
                candidates.append(cfg)
                local_cfg = os.path.join(self.model_artifact_dir, "config.json")
                if not os.path.exists(local_cfg):
                    try:
                        shutil.copy2(cfg, local_cfg)
                    except OSError:
                        pass
        local_cfg = os.path.join(self.model_artifact_dir, "config.json")
        if os.path.exists(local_cfg):
            candidates.append(local_cfg)

        for cfg_path in candidates:
            self._patch_rope_scaling(cfg_path)

    def _patch_rope_scaling(self, config_path: str) -> None:
        rope_scaling = self._load_rope_scaling_from_file(config_path)
        if not rope_scaling:
            return
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return

        config["rope_scaling"] = rope_scaling
        try:
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(config, fh)
        except OSError:
            return

    def _load_rope_scaling_from_file(self, config_path: str):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

        rope_scaling = config.get("rope_scaling")
        return self._normalize_rope_scaling(rope_scaling)

    @staticmethod
    def _normalize_rope_scaling(rope_scaling):
        if isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
            if "type" not in rope_scaling:
                rope_type = rope_scaling.get("rope_type")
                if rope_type:
                    rope_scaling["type"] = rope_type
        return rope_scaling

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        # The chat template ignores the system prompt;
        conversation = []
        if sys_prompt_arg:
            conversation = [{"role": "system", "content": sys_prompt_arg}]

        conversation += [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1
        )

        response = self.client.generate([chat_str], sampling_params)
        response = response[0].outputs[0].text

        self._save_cache(sys_prompt_arg + prompt, response)

        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
