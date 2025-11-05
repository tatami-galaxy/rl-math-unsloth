import os
from os.path import dirname
import chz
from functools import partial

from experiments.sft.sft_config import SFTHyps
from template import SYSTEM_PROMPT, create_chat_template

from datasets import load_dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Disable standby mode to avoid expandable_segments conflict
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"


def get_root_dir():
    root = os.path.abspath('')
    while root.split('/')[-1] != 'rl-math-unsloth':
        root = dirname(root)
    return root

# format dataset for sft
def format_dataset(x, tokenizer):
    problem = x["problem"]
    solution = x["generated_solution"]
    expected_answer = x["expected_answer"]
    messages =  [
        {"role" : "system",    "content" : SYSTEM_PROMPT},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : solution},
    ]
    return {
        "text" : tokenizer.apply_chat_template(messages, tokenize=False),
        "expected_answer" : expected_answer,
    }



def main(config: SFTHyps):

    root = get_root_dir()

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model_name,
        max_seq_length = config.max_seq_len,
        load_in_4bit = config.load_in_4bit,
        load_in_8bit = config.load_in_8bit,
        load_in_16bit = config.load_in_16bit,
        fast_inference = config.fast_inference, 
        max_lora_rank = config.lora_rank,
        gpu_memory_utilization = config.gpu_memory_utilization
    )
    print(tokenizer.chat_template)
    # Load LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = config.lora_rank*2 if config.lora_alpha is None else config.lora_alpha, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = config.seed,
    )

    # create or modify chat template
    tokenizer = create_chat_template(tokenizer)

    # Load dataset
    dataset = load_dataset(config.sft_dataset, split=config.sft_dataset_split)

    # Filter None values
    dataset = dataset.filter(lambda x: x["expected_answer"] is not None)

    # Format dataset to chat template
    dataset = dataset.map(
        partial(
            format_dataset,
            tokenizer=tokenizer,
        ),
        batched = False,
        remove_columns=dataset.column_names,
    )

    # Truncate fine-tuning dataset to max_seq_len
    dataset = dataset.filter(lambda x: len(x["text"]) <= config.max_seq_len)

    # Train
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = config.per_device_train_batch_size,
            gradient_accumulation_steps = config.gradient_accumulation_steps, # Use GA to mimic batch size!
            warmup_steps = config.warmup_steps,
            num_train_epochs = config.num_train_epochs, # Set this for 1 full training run.
            learning_rate = config.learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = config.logging_steps,
            #optim = config.optim,
            weight_decay = config.weight_decay,
            lr_scheduler_type = config.lr_scheduler_type,
            seed = config.seed,
            report_to = "tensorboard", # Use TrackIO/WandB etc
            output_dir=root+"/"+config.output_dir,
        ),
    )

    trainer.train()



if __name__ == "__main__":
    chz.nested_entrypoint(main)
