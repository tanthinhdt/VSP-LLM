from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, is_dataclass
from omegaconf import OmegaConf, MISSING, II


@dataclass
class VSP_LLM_TrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={
            "help": "label frame rate. -1 for sequence label"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    llm_ckpt_path: str = field(
        default=MISSING,
        metadata={
            "help": "path to llama checkpoint"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={
            "help": "pad shorter samples instead of cropping"
        },
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "max sample size to keep in training"
        },
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "min sample size to keep in training"
        },
    )
    max_trim_sample_size: Optional[int] = field(
        default=II("task.max_sample_size"),
        metadata={
            "help": "max sample size to trim to for batching"
        },
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={
            "help": "always crop from the beginning if false"
        },
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={
            "help": "pad audio to the longest one in the batch if true"
        },
    )
    pdb: Optional[bool] = field(
        default=False,
        metadata={
            "help": "pdb"
        },
    )
    stack_order_audio: int = field(
        default=1,
        metadata={
            "help": "concatenate n consecutive audio frames for one step"
        },
    )
    skip_verify: Optional[bool] = field(
        default=False,
        metadata={
            "help": "skip verifying label-audio alignment"
        },
    )
    image_aug: bool = field(
        default=False,
        metadata={
            "help": "image data augmentation"
        }
    )
    image_crop_size: int = field(
        default=88,
        metadata={
            "help": "image ROI size"
        }
    )
    image_mean: float = field(
        default=0.421,
        metadata={
            "help": "image mean"
        }
    )
    image_std: float = field(
        default=0.165,
        metadata={
            "help": "image std"
        }
    )
    modalities: Optional[List[str]] = field(
        default_factory=lambda: ["audio", "video"],
        metadata={"help": "modalities to load"},
    )
    is_s2s: bool = field(
        default=False,
        metadata={
            "help": "seq2seq fine-tuning only"
        }
    )
    tokenizer_bpe_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "tokenizer model name"
        }
    )
    tokenizer_bpe_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "tokenizer model path"
        }
    )
    noise_wav: Optional[str] = field(
        default=None,
        metadata={
            "help": "manifest of noise wav files (one wav file path per line)"
        },
    )
    noise_prob: float = field(
        default=0,
        metadata={
            "help": "noise probability"
        }
    )
    noise_snr: Optional[str] = field(
        default="0",
        metadata={
            "help": "noise SNR in audio"
        }
    )
    noise_num: int = field(
        default=1,
        metadata={
            "help": "number of noise wav files to mix"
        }
    )
    fine_tuning: bool = field(
        default=False,
        metadata={
            "help": "set to true if fine-tuning AV-Hubert"
        }
    )


@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(
        default=None,
        metadata={"help": "noise wav file"},
    )
    noise_prob: float = field(
        default=0,
        metadata={"help": "noise probability"},
    )
    noise_snr: float = field(
        default=0,
        metadata={"help": "noise SNR in audio"},
    )
    modalities: List[str] = field(
        default_factory=lambda: ["video"],
        metadata={"help": "which modality to use"},
    )
    data: Optional[str] = field(
        default=None,
        metadata={"help": "path to test data directory"},
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={"help": "path to test label directory"},
    )
    eval_bleu: bool = field(
        default=False,
        metadata={"help": "evaluate bleu score"},
    )
    llm_ckpt_path: str = field(
        default=MISSING,
        metadata={"help": "path to llama checkpoint"},
    )


@dataclass
class GenerationConfig(FairseqDataclass):
    """
    For more details, please visit:
    https://huggingface.co/docs/transformers/main_classes/text_generation
    """
    max_length: int = field(default=20)
    max_new_tokens: Optional[int] = field(default=None)
    min_length: int = field(default=0)
    min_new_tokens: Optional[int] = field(default=None)
    max_time: Optional[float] = field(default=None)

    do_sample: bool = field(default=False)
    num_beams: int = field(default=1)
    num_beam_groups: int = field(default=1)

    temperature: float = field(default=1.0)
    top_k: float = field(default=50)
    top_p: float = field(default=1.0)
    min_p: Optional[float] = field(default=None)
    typical_p: float = field(default=1.0)
    epsilon_cutoff: float = field(default=0.0)
    eta_cutoff: float = field(default=0.0)
    diversity_penalty: float = field(default=0.0)
    repetition_penalty: float = field(default=1.0)
    encoder_repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    no_repeat_ngram_size: float = field(default=0)
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = field(default=None)


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": (
                "if true, assumes we are using ax for tuning"
                "and returns a tuple for ax to consume"
            )
        },
    )
    show_sample: bool = field(
        default=False,
        metadata={
            "help": "Show ref, hypo and evaluation results of each sample"
        }
    )
