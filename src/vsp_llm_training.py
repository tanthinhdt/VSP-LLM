# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import numpy as np
from typing import List, Optional, Tuple
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from src.configs import VSP_LLM_TrainingConfig


DBG = True if len(sys.argv) == 1 else False

if DBG:
    from vsp_llm_dataset import VSP_LLM_dataset
else:
    from .vsp_llm_dataset import VSP_LLM_dataset

logger = logging.getLogger(__name__)


@register_task("vsp_llm_training", dataclass=VSP_LLM_TrainingConfig)
class VSP_LLM_TrainingTask(FairseqTask):

    cfg: VSP_LLM_TrainingConfig

    def __init__(
        self,
        cfg: VSP_LLM_TrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"AVHubertPretrainingTask Config {cfg}")

        self.fine_tuning = cfg.fine_tuning
        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def dictionaries(self) -> List[Dictionary]:
        return None

    @classmethod
    def setup_task(
        cls, cfg: VSP_LLM_TrainingConfig, **kwargs
    ) -> "Avhubert_Llama_Cluster_Trans_PretrainingTask":
        if cfg.pdb:
            import pdb

            pdb.set_trace()
        return cls(cfg)

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        logger.info("Using tokenizer")
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
        image_aug = self.cfg.image_aug if split == "train" else False
        noise_fn, noise_snr = (
            f"{self.cfg.noise_wav}/{split}.tsv"
            if self.cfg.noise_wav is not None
            else None
        ), eval(self.cfg.noise_snr)
        noise_num = self.cfg.noise_num  #
        self.datasets[split] = VSP_LLM_dataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            llm_ckpt_path=self.cfg.llm_ckpt_path,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=True,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            noise_fn=noise_fn,
            noise_prob=self.cfg.noise_prob,
            noise_snr=noise_snr,
            noise_num=noise_num,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        return indices
