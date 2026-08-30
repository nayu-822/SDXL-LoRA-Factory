from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DatasetSyncRequest(BaseModel):
    gdrive_path: str = Field(..., min_length=1)
    dataset_name: Optional[str] = None


class CaptionSyncRequest(BaseModel):
    local_dataset_path: str = Field(..., min_length=1)
    gdrive_path: str = Field(..., min_length=1)


class OutputSyncRequest(BaseModel):
    local_output_path: str = Field(..., min_length=1)
    gdrive_path: str = Field(..., min_length=1)


class TagUpdate(BaseModel):
    path: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)


class BatchTagUpdate(BaseModel):
    path: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    position: str = "append"


class TriggerWordUpdate(BaseModel):
    path: str = Field(..., min_length=1)
    trigger_word: str = Field(..., min_length=1)


class TaggerConfig(BaseModel):
    path: str = Field(..., min_length=1)
    model: str = ""
    auto_sync_captions: bool = False
    gdrive_path: str = ""
    overwrite_existing_captions: bool = False


class RecommendationRequest(BaseModel):
    image_count: int = Field(..., ge=0)
    training_type: str = "character"
    vram: str = "balanced"
    gradient_accumulation_steps: int = Field(1, ge=1, le=128)


class TrainingConfig(BaseModel):
    # Compatibility fields from the Windows v4.x UI.
    path: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    vae: str = ""
    output_dir: str = ""
    name: str = "my_sdxl_lora"
    training_type: str = "character"
    vram: str = "balanced"
    trigger_word: str = ""
    epochs: int = Field(10, ge=1, le=10000)
    lr: float = Field(1e-4, gt=0)
    rank: int = Field(16, ge=1, le=1024)
    alpha: float = Field(16, gt=0, le=1024)

    # Dataset and training controls.
    repeats: int = Field(10, ge=1, le=10000)
    batch_size: int = Field(1, ge=1, le=256)
    resolution_width: int = Field(1024, ge=64, le=4096)
    resolution_height: int = Field(1024, ge=64, le=4096)
    min_bucket_reso: int = Field(256, ge=64, le=4096)
    max_bucket_reso: int = Field(2048, ge=64, le=8192)
    gradient_accumulation_steps: int = Field(1, ge=1, le=128)
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    workers: int = Field(2, ge=0, le=64)
    seed: Optional[int] = None
    unet_lr: Optional[float] = Field(None, gt=0)
    text_encoder_lr: Optional[float] = Field(None, gt=0)

    # Existing advanced settings.
    flip_aug: bool = False
    shuffle_caption: bool = False
    keep_tokens: int = Field(1, ge=0, le=128)
    min_snr_gamma: bool = False
    min_snr_gamma_value: float = Field(5.0, ge=0)
    optimizer_type: str = "AdamW"
    optimizer_args: str = ""
    optimizer_low_vram: bool = False
    keep_unet: bool = False

    # Sampling settings passed to sd-scripts.
    sample_prompt: str = ""
    sample_prompts: List[str] = Field(default_factory=list)
    sample_negative_prompt: str = ""
    sample_seed: int = 12345
    sample_sampler: str = "euler_a"
    sample_steps: int = Field(20, ge=1, le=1000)
    sample_width: int = Field(1024, ge=64, le=4096)
    sample_height: int = Field(1024, ge=64, le=4096)
    sample_scale: float = Field(7.0, gt=0, le=100)
    sample_every_n_epochs: int = Field(1, ge=0, le=10000)
    sample_at_first: bool = True

    # Drive integration.
    gdrive_dataset_path: str = ""
    gdrive_output_path: str = ""
    auto_sync_output: bool = False

    # Kept for Windows compatibility. RunPod deliberately ignores shutdown.
    shutdown: bool = False

    class Config:
        extra = "ignore"


class StatusQuery(BaseModel):
    job_id: Optional[str] = None
