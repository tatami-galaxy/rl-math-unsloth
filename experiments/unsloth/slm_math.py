import os
import argparse

import torch

from unsloth import FastLanguageModel


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
    )
    parser.add_argument(
        "--sft-dataset",
        type=str,
    )
    parser.add_argument(
        "--sft",
        action="store_true",
    )
    # generation
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
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
    return parser.parse_args()


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

    
    
if __name__ == "__main__":
    main()
