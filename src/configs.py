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
from omegaconf import OmegaConf, MISSING


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
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )
