"""RGB IKV variant of the official DM05-MEM RoboDojo cover_blocks entry."""

from dataclasses import dataclass, field

import tyro

from playground.dm05_mem_sft_robodojo_cover_blocks import (
    DM05Exp as _DM05Exp,
    DM05ModelConfig as _DM05ModelConfig,
    DM05TrainerConfig as _DM05TrainerConfig,
)


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    ikv_rgb_enabled: bool = True
    ikv_history_capacity: int = 320
    ikv_top_k: int = 320
    ikv_motion_only: bool = False
    ikv_motion_threshold: float = 0.04
    ikv_dino_model_path: str | None = None
    ikv_require_dino: bool = True


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = "user_checkpoints/dm05_mem_ikv_rgb_robodojo_cover_blocks"


@dataclass
class DM05Exp(_DM05Exp):
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
