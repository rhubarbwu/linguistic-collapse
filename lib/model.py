import json
import os
from argparse import Namespace
from dataclasses import dataclass, field
from logging import Logger
from typing import List, Optional, Tuple

import numpy as np
import torch as pt
from torch import Tensor
from torch.nn import Identity
from transformers import (CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,
                          AutoConfig)
from transformers import AutoModelForCausalLM as AutoCLM
from transformers import AutoTokenizer

from lib.utils import identify

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


def get_config(args: ModelArguments, logger: Logger) -> AutoConfig:
    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.config_overrides is not None:
            logger.info(f"Overriding config: {args.config_overrides}")
            config.update_from_string(args.config_overrides)
            logger.info(f"New config: {config}")
    return config


def get_tokenizer(args: ModelArguments) -> AutoTokenizer:
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    return tokenizer


def select_ckpt(path: str, idx: int) -> Optional[str]:
    dir_files = os.listdir(path)
    ckpt_files = [f for f in dir_files if f.startswith(f"checkpoint-")]
    epoch_paths = sorted(ckpt_files, key=lambda s: int(s.split("-")[1]))
    if idx < len(epoch_paths):
        return f"{path}/{epoch_paths[idx]}"


def get_model(
    args: ModelArguments,
    config: AutoConfig,
    logger: Logger = None,
    model_ckpt_idx: int = None,
) -> AutoCLM:
    if args.model_name_or_path:
        path = args.model_name_or_path
        if model_ckpt_idx:
            path = select_ckpt(path, model_ckpt_idx)

        torch_dtype = (
            args.torch_dtype
            if args.torch_dtype in ["auto", None]
            else getattr(pt, args.torch_dtype)
        )
        model = AutoCLM.from_pretrained(
            path,
            from_tf=bool(".ckpt" in path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        model = AutoCLM.from_config(config, trust_remote_code=args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        if logger:
            logger.info(
                f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
            )

    return model


def split_parts(model_iden_or_name: str) -> Tuple[int, int, int]:
    iden = identify(model_iden_or_name)
    depth_str, width_ckpt_str = iden.split("x")
    width_str, ckpt_str = width_ckpt_str.split("@")

    return int(depth_str), int(width_str), int(ckpt_str)


def strip_model(model: AutoCLM, device: str = "cpu") -> Tuple[AutoCLM, int, int]:
    C, D = model.lm_head.out_features, model.lm_head.in_features
    del model.lm_head
    model.lm_head = Identity()
    model.to(device)
    return model, C, D


def get_model_loss(model_name: str, args: Namespace) -> Optional[np.float64]:
    model_path = f"{args.model_cache}/{model_name}"
    trainer_state_file = f"{model_path.split('@')[0]}/trainer_state.json"
    if not os.path.exists(trainer_state_file):
        return None

    _, _, idx = split_parts(model_name)
    with open(trainer_state_file, "r") as f:
        log = json.load(f)["log_history"]

    epoch_losses = [step["loss"] for step in log if idx <= step["epoch"] < idx + 1]
    epoch_avg_loss = np.mean(epoch_losses).astype(np.float64)
    return epoch_avg_loss


def get_classifier_weights(model_name: str, args: Namespace) -> Optional[Tensor]:
    model_path = f"{args.model_cache}/{model_name}"
    classifier_file = f"{args.model_cache}/cls-{model_name}.pt"
    if os.path.exists(classifier_file):
        return pt.load(classifier_file, args.device)

    print(f"classifier weights file for {classifier_file} not found...")
    if args.force_load or input("force load? ").upper().startswith("Y"):
        _, _, ckpt_idx = split_parts(model_name)
        ckpt_path = select_ckpt(model_path.split("@")[0], ckpt_idx)
        if ckpt_path is None:
            print(f"W: model checkpoint at index {ckpt_idx} not found")
            return None

        model = AutoCLM.from_pretrained(
            ckpt_path,
            from_tf=bool(".ckpt" in ckpt_path),
        )
        W = list(model.lm_head.parameters())[0]
        del model

        print(
            f"caching weights for {ckpt_path} in {classifier_file}",
            flush=True,
        )
        pt.save(W.detach(), classifier_file)

        return W.to(args.device)
