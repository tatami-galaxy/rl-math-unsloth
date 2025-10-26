import os
import argparse
import gc
import pandas as pd
import numpy as np

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from datasets import load_dataset, Dataset


os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:False"


def parse_arguments():

    parser = argparse.ArgumentParser(description="RL Math experiments with Unsloth")
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="open-r1/DAPO-Math-17k-Processed",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--sft-dataset",
        type=str,
        default="unsloth/OpenMathReasoning-mini"
    )
    parser.add_argument(
        "--sft-dataset-split",
        type=str,
        default="cot"
    )
    parser.add_argument(
        "--sft",
        action="store_true",
    )
    # lora
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
    )
    # vllm
    parser.add_argument(
        "--fast-inference",
        action="store_true",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.9,
    )
    # training
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5   
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4   
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5   
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit"   
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01   
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear"   
    )
    # generation
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
    ) 
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6 
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95 
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20 
    )
    return parser.parse_args()



def create_chat_template(tokenizer):
    reasoning_start = "<think>"
    reasoning_end   = "</think>"  

    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    system_prompt = "You are given a math problem. Please reason step by step, and put your final answer within \\boxed{}."
    
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    
    tokenizer.chat_template = chat_template

    return tokenizer, system_prompt



def format_dataset(x, system_prompt):
    problem = x["problem"]
    solution = x["generated_solution"]
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : solution},
    ]



def extract_hash_answer(text):
    # if "####" not in text: return None
    # return text.split("####")[1].strip()
    return text


def main():

    # parse arguments
    args = parse_arguments()

    # load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_len,
        load_in_4bit = args.load_in_4bit, # False for LoRA 16bit
        fast_inference = args.fast_inference, # Enable vLLM fast inference
        max_lora_rank = args.lora_rank,
        gpu_memory_utilization = args.gpu_mem_util, # Reduce if out of memory
    )
    # load lora
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = args.lora_rank*2 if args.lora_alpha is None else args.lora_alpha, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = args.seed,
    )

    # create or modify chat template
    tokenizer, system_prompt = create_chat_template(tokenizer)

    # sft before rl to force format
    if args.sft:

        # load sft dataset
        dataset = load_dataset(args.sft_dataset, split=args.sft_dataset_split)
        # convert to pandas
        dataset = dataset.to_pandas()[
            ["expected_answer", "problem", "generated_solution"]
        ]

        # try converting answer to number - if not, replace with NaN
        is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
        # select samples with number as expected answer
        dataset = dataset.iloc[np.where(is_number)[0]]

        # format dataset
        dataset["messages"] = dataset.apply(format_dataset, args=(system_prompt,), axis=1)
        #tokenizer.apply_chat_template(dataset["messages"][0], tokenize=False)

        # truncate fine-tuning dataset to max_seq_len/2
        dataset["N"] = dataset["messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
        dataset = dataset.loc[dataset["N"] <= args.max_seq_len/2].copy()
        #print(dataset.shape)

        # hf dataset
        dataset["text"] = tokenizer.apply_chat_template(dataset["messages"].values.tolist(), tokenize=False)
        dataset = Dataset.from_pandas(dataset)
        #print(dataset)

        # sft training
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
                output_dir='./results'
            ),
        )

        trainer.train()

        """print("Testing")
        print('')
        text = tokenizer.apply_chat_template(
            dataset[0]["messages"][:2],
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
        )
        print(model.generate(
            **tokenizer(text, return_tensors = "pt").to("cuda"),
            temperature = 0.6,
            top_p = 0.95,
            top_k = 20,
            do_sample = True,
            max_new_tokens = args.max_seq_len,
            streamer = TextStreamer(tokenizer, skip_prompt = False),
        ))"""

        # clean up
        del dataset
        torch.cuda.empty_cache()
        gc.collect()

    # rl
    dataset = load_dataset(args.dataset, args.dataset_config, split = "train")

    # process dataset
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    }) 
    print(dataset[0])

    # extract from boxed, maybe math-verify




    
    
if __name__ == "__main__":
    main()
