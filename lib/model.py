import os
from argparse import Namespace
from dataclasses import dataclass, field
from logging import Logger
from typing import List, Optional, Tuple

import torch as pt
from torch import Tensor
from torch.nn import Identity
from transformers import (CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer)

from lib.visualization import get_shade

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

COLOUR_BASES = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

DEPTHS = [1, 2, 4, 8, 12]
WIDTHS = [64, 128, 256, 512, 768, 1024]


def get_model_colour(
    depth: str, width: str, depths: List[int], widths: List[int], by_width: bool = False
) -> str:
    idx_d, idx_w = depths.index(int(depth)), widths.index(int(width))

    base = COLOUR_BASES[idx_w if by_width else idx_d]
    shade_idx = idx_d if by_width else idx_w
    shade_count = len(depths if by_width else widths)
    shade = get_shade(base, shade_idx, shade_count)
    return shade


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


def get_model(
    args: ModelArguments, config: AutoConfig, logger: Logger
) -> AutoModelForCausalLM:
    if args.model_name_or_path:
        torch_dtype = (
            args.torch_dtype
            if args.torch_dtype in ["auto", None]
            else getattr(pt, args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=args.trust_remote_code
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )

    return model


def get_classifier_weights(args: Namespace) -> Tensor:
    classifier_file = f"{args.model_cache}/{args.model_name}-cls.pt"
    if not os.path.exists(classifier_file):
        print(f"classifier weights file for {classifier_file} not found...")
        return None

    return pt.load(classifier_file, args.device)


def strip_model(
    args: ModelArguments, model: AutoModelForCausalLM, device: str = "cpu"
) -> Tuple[AutoModelForCausalLM, int, int]:
    C, D = model.lm_head.out_features, model.lm_head.in_features
    W = list(model.lm_head.parameters())[0].detach()
    del model.lm_head
    model.lm_head = Identity()

    classifier_file = f"{args.model_name_or_path}-cls.pt"
    if not os.path.exists(classifier_file):
        print(
            f"caching weights for {args.model_name_or_path} in {classifier_file}",
            flush=True,
        )
        pt.save(W, classifier_file)

    model.to(device)
    return model, C, D
