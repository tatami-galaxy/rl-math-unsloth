import chz

@chz.chz
class RLHyps:

    seed: int = 42

    # Dataset
    # math-dataset/DeepScaleR-Preview-Dataset
    # POLARIS-Project/Polaris-Dataset-53K
    # open-r1/DAPO-Math-17k-Processed
    rl_dataset: str = "open-r1/DAPO-Math-17k-Processed"
    rl_dataset_config: str = "en"

    # Model parameters
    model_name: str = None
    lora_config: str = None
    load_checkpoint_path: str | None = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_16bit: bool = True
    max_seq_len: int = 4096

    # Training parameters
    temperature: float = 1.0
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1 # Increase to 4 for smoother training
    num_generations: int = 8 # Decrease if out of memory
    num_train_epochs: int = 1 # Set to 1 for a full training run
    max_steps: int = 100
    save_steps: int = 100
    fp16: bool = False


    # vLLM parameters
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.9
    min_p: float = 0.1
    top_p: float = 1.0
    top_k: int = -1

    # LoRA parameters
    lora_rank: int = 32
    lora_alpha: int | None = None

    # Checkpointing and evaluation
    save_every: int = 20
    eval_every: int = 10

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    output_dir: str = "/models/rl_checkpoints"
    log_path: str = "./rl_logs"