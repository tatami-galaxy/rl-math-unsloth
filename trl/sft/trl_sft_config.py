from dataclasses import dataclass, field


@dataclass
class TRLSFTHyps:

    seed: int = 42

    # Dataset
    sft_dataset: str = field(default="zwhe99/DeepMath-103K")
    sft_dataset_split: str = field(default="train")
    sample: bool = field(default=False)
    num_samples: int = field(default=512)

    # Model parameters
    model_name: str = field(default=None)
    model_revision: str = field(default="main")

    # Training parameters
    max_seq_len: int = field(default=None)     # 4096, 8192, 16384
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    pad_to_multiple_of: int = field(default=4)
    num_train_epochs: int = field(default=1)

    # Checkpointing and evaluation
    logging_steps: int = field(default=8)
    save_steps: int = field(default=1000)

    # Optimizer parameters
    optim: str = field(default="adamw_torch_fused")
    learning_rate: float = field(default=2e-5)     # 1e-4, 2e-5 
    warmup_steps: int = field(default=8)
    lr_scheduler_type: str = field(default="constant")   # linear, constant
    weight_decay: float = field(default=0.01)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_eps: float = field(default=1e-8)

    # Logging parameters
    output_dir: str = field(default="/models/trl/sft_checkpoints")