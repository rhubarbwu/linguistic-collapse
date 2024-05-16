import json
import os
from argparse import Namespace
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, Optional, Tuple

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

    name_split: str = field(
        default="x",
        metadata={
            "help": ("Which group of models to analyze."),
            "choices": ["x", "y", "z"],
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
    tknzer_args = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tokenizer_name:
        tknzer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tknzer_args)
    elif args.model_name_or_path:
        tknzer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tknzer_args)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    return tknzer


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
        if model_ckpt_idx != None:
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
    depth_str, rest_str = iden.split("x")
    width_str, rest_str = rest_str.split("_")
    opt_str, ckpt_str = rest_str.split("@")

    return int(depth_str), int(width_str), int(ckpt_str)


def strip_model(
    model: AutoCLM, device: str = "cpu"
) -> Tuple[int, int, AutoCLM, Tensor]:
    C, D = model.lm_head.out_features, model.lm_head.in_features
    W = list(model.lm_head.parameters())[0].detach()
    del model.lm_head
    model.lm_head = Identity()
    return C, D, model.to(device), W.to(device)


def get_model_stats(model_name: str, args: Namespace) -> Dict[str, np.number]:
    model_path, ckpt_idx = f"{args.model_cache}/{model_name}".split("@")
    model_stats, trained_prop = {}, None

    config_file = f"{model_path}/config.json"
    if os.path.exists(config_file):
        config = AutoConfig.from_pretrained(config_file)
        model = AutoCLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        model_stats["n_params"] = int(n_params)

    # results_file = f"{model_path}/train_results.json"
    # if os.path.exists(results_file):
    #     with open(results_file, "r") as f:
    #         data = json.load(f)
    #     trained_prop = (int(ckpt_idx) + 1) / data["epoch"]
    #     model_stats["train_time"] = np.float64(data["train_runtime"] * trained_prop)

    # state_file = f"{model_path}/trainer_state.json"
    # if os.path.exists(state_file):
    #     _, _, idx = split_parts(model_name)
    #     with open(state_file, "r") as f:
    #         data = json.load(f)
    #     log = data["log_history"]
    #     epoch_losses = [step["loss"] for step in log if idx <= step["epoch"] < idx + 1]
    #     model_stats["train_loss"] = np.mean(epoch_losses).astype(np.float64)
    #     if trained_prop is not None:
    #         model_stats["train_flops"] = np.int64(data["total_flos"] * trained_prop)

    model_path = f"{args.model_cache}/{model_name}".split("@")[0]
    _, _, ckpt_idx = split_parts(model_name)
    ckpt_path = select_ckpt(model_path, ckpt_idx)
    if ckpt_path is None:
        print(f"W: model checkpoint at index {ckpt_idx} not found")
        return None

    eval_file = f"{ckpt_path}/eval_results.json"
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            data = json.load(f)
        model_stats["val_acc"] = data["eval_accuracy"]
        model_stats["val_loss"] = data["eval_loss"]
        model_stats["perplex"] = data["perplexity"]

    return model_stats


def get_classifier_weights(model_name: str, args: Namespace) -> Optional[Tensor]:
    model_path = f"{args.model_cache}/{model_name}".split("@")[0]
    _, _, ckpt_idx = split_parts(model_name)
    ckpt_path = select_ckpt(model_path, ckpt_idx)
    if ckpt_path is None:
        print(f"W: model checkpoint at index {ckpt_idx} not found")
        return None

    model = AutoCLM.from_pretrained(
        ckpt_path,
        from_tf=bool(".ckpt" in ckpt_path),
    )
    W = list(model.lm_head.parameters())[0]
    del model

    return W.to(args.device)
