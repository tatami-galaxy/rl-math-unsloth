import chz

@chz.chz
class SFTHyps:

    seed: int = 42

    # Dataset
    sft_dataset: str = "zwhe99/DeepMath-103K"
    sft_dataset_split: str = "train"

    # Model parameters
    model_name: str # required
    load_checkpoint_path: str | None = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_16bit: bool = False
    max_seq_len: int = 8192

    # Training parameters
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    warmup_steps: int = 5
    logging_steps: int = 5
    optim: str = "adamw_torch_fused"
    weight_decay: float = 0.01
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear"

    # vLLM parameters
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.9
    device_map: str = "balanced"

    # LoRA parameters
    lora_rank: int = 64
    lora_alpha: int | None = None

    # Checkpointing and evaluation
    save_every: int = 20
    eval_every: int = 10

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    output_dir: str = "/models/sft_checkpoints"
    log_path: str = "./sft_logs"