import os
import sys
sys.path.append("..")
from trl.sft.sft_config import SFTHyps
from utils import SYSTEM_PROMPT, REASONING_START
from utils import get_root_dir, create_chat_template
quit()

from functools import partial
import chz

from datasets import load_dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# causes expandable_segments conflict
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"


# format dataset for sft
def format_dataset(x, tokenizer):

    def process_trace(q, trace):
        messages =  [
            {"role" : "system",    "content" : SYSTEM_PROMPT},
            # add <think> token after question
            {"role" : "user",      "content" : q+REASONING_START},
            {"role" : "assistant", "content" : trace},
        ]
        # no generation_prompt because this is sft, needed for rl
        return tokenizer.apply_chat_template(messages, tokenize=False) 

    new_examples = {
        "text": [],
        "answer": [],
    }
    # all 3 traces
    for i, question in enumerate(x["question"]):
        answer = x["final_answer"][i]
        traces = [x["r1_solution_1"][i], x["r1_solution_2"][i], x["r1_solution_3"][i]]
        for trace in traces:
            new_examples["text"].append(process_trace(question, trace))
            new_examples["answer"].append(answer)

    return new_examples



def main(config: SFTHyps):

    root = get_root_dir()

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model_name,
        max_seq_length = config.max_seq_len,
        load_in_16bit = config.load_in_16bit, 
        load_in_8bit = config.load_in_8bit, 
        load_in_4bit = config.load_in_4bit, 
        full_finetuning = config.full_finetuning, 
        fast_inference = config.fast_inference, 
        max_lora_rank = config.lora_rank,
        gpu_memory_utilization = config.gpu_memory_utilization,
        device_map = config.device_map,
    )
    if not config.full_finetuning:
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

    # load dataset
    dataset = load_dataset(config.sft_dataset, split=config.sft_dataset_split)

    # format dataset to chat template
    # create 3 examples from each deepmath example
    # with the 3 given reasoning traces
    dataset = dataset.map(
        partial(
            format_dataset,
            tokenizer=tokenizer,
        ),
        batched = True,
        remove_columns=dataset.column_names,
    ).shuffle(config.seed)

    # truncate fine-tuning dataset to max_seq_len
    # seq_len: 8192 -> dataset: 62255
    # seq_len: 4096 -> dataset: 1877
    dataset = dataset.filter(lambda x: len(x["text"]) <= config.max_seq_len)

    # set output directory
    model_name = config.model_name.split("/")[-1]
    dataset_name = config.sft_dataset.split("/")[-1]
    if config.full_finetuning:
        train_type = "full"
    else:
        train_type = "lora_{}".format(config.lora_rank)
    seq_len = config.max_seq_len
    checkpoint_folder = model_name + "_" + dataset_name + "_" + train_type + "_" + str(seq_len)
    output_dir = root+"/"+config.output_dir+"/"+checkpoint_folder
    
    # Train
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = config.per_device_train_batch_size,
            gradient_accumulation_steps = config.gradient_accumulation_steps, # Use GA to mimic batch size
            warmup_steps = config.warmup_steps,
            num_train_epochs = config.num_train_epochs, # Set this for 1 full training run.
            learning_rate = config.learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = config.logging_steps,
            save_steps = config.save_steps,
            optim = config.optim,
            weight_decay = config.weight_decay,
            lr_scheduler_type = config.lr_scheduler_type,
            seed = config.seed,
            report_to = "tensorboard", 
            output_dir=output_dir
        ),
    )

    trainer.train()



if __name__ == "__main__":
    chz.nested_entrypoint(main)
