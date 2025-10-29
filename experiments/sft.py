import os
import chz
from functools import partial

from sft_config import SFTHyps
from template import create_chat_template

from datasets import load_dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Disable standby mode to avoid expandable_segments conflict
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# format dataset for sft
def format_dataset(x, system_prompt, tokenizer):
    problem = x["problem"]
    solution = x["generated_solution"]
    expected_answer = x["expected_answer"]
    messages =  [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : solution},
    ]
    return {
        "text" : tokenizer.apply_chat_template(messages, tokenize=False),
        "expected_answer" : expected_answer,
    }



def main(config: SFTHyps):

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
    tokenizer, system_prompt = create_chat_template(tokenizer)

    # Load dataset
    dataset = load_dataset(config.sft_dataset, split=config.sft_dataset_split)

    # Filter None values
    dataset = dataset.filter(lambda x: x["expected_answer"] is not None)

    # Format dataset to chat template
    dataset = dataset.map(
        partial(
            format_dataset,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        ),
        batched = False,
        remove_columns=dataset.column_names,
    )

    # Truncate fine-tuning dataset to max_seq_len
    dataset = dataset.filter(lambda x: len(x["text"]) <= config.max_seq_len)
    print(dataset)
    quit()

    # Train
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps, # Use GA to mimic batch size!
            warmup_steps = args.warmup_steps,
            num_train_epochs = args.num_train_epochs, # Set this for 1 full training run.
            learning_rate = args.learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = args.logging_steps,
            optim = args.optim,
            weight_decay = args.weight_decay,
            lr_scheduler_type = args.lr_scheduler_type,
            seed = args.seed,
            report_to = "tensorboard", # Use TrackIO/WandB etc
            output_dir='./sft_runs'
        ),
    )

    trainer.train()



if __name__ == "__main__":
    chz.nested_entrypoint(main)
