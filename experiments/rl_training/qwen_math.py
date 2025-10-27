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
from math_verify import parse, verify


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

    # extract from boxed, maybe math-verify

    text = """
                <think>
            Okay, so I need to find the expected value of X, E(X), given the joint probability generating function φ(s, t) = (1 - α - β)/(1 - α s - β t). Hmm, I remember that probability generating functions (PGFs) are useful for finding moments like the expectation. But this is a joint PGF for two variables, X and Y, right? So, maybe I need to find the marginal PGF for X first and then take its derivative at s=1 to get E(X). Let me recall the steps.

            First, the joint PGF φ(s, t) is defined as E[s^X t^Y]. To get the marginal PGF for X, I should set t = 1 in φ(s, t). That makes sense because setting t to 1 would sum over all possible Y values, leaving just the generating function for X. So, the marginal PGF for X would be φ(s, 1). Let me compute that.

            Substituting t = 1 into the given function: φ(s, 1) = (1 - α - β)/(1 - α s - β * 1) = (1 - α - β)/(1 - β - α s). Okay, so that simplifies to (1 - α - β)/( (1 - β) - α s ). Maybe I can factor out (1 - β) in the denominator? Let's see:

            Denominator: (1 - β) - α s = (1 - β)(1 - (α/(1 - β)) s). So then the PGF becomes (1 - α - β)/[(1 - β)(1 - (α/(1 - β)) s)] = [ (1 - α - β)/(1 - β) ] * [1 / (1 - (α/(1 - β)) s) ].

            Hmm, the term [1 / (1 - (α/(1 - β)) s)] looks like the PGF of a geometric distribution. The standard PGF for a geometric distribution with parameter p is p/(1 - (1 - p)s), but here the numerator is different. Wait, actually, maybe it's a shifted geometric distribution? Let me check. If the PGF is [something] / (1 - something*s), then the coefficients would correspond to probabilities multiplied by s^k. For it to be a valid PGF, the numerator must be such that when expanded as a power series in s, all coefficients are non-negative and sum to 1.

            But maybe I don't need to find the distribution explicitly. Instead, to find E(X), I can take the first derivative of the marginal PGF with respect to s, evaluated at s=1. That's the standard method. The expectation E(X) is the first derivative of the PGF at s=1.

            So let me compute d/ds [φ(s,1)] evaluated at s=1.

            First, φ(s,1) = (1 - α - β)/(1 - β - α s). Let's compute the derivative with respect to s.

            Let me denote D = denominator = 1 - β - α s. Then φ(s,1) = (1 - α - β)/D.

            The derivative of φ with respect to s is (1 - α - β) * d/ds [1/D] = (1 - α - β) * [ derivative of (D^{-1}) ].

            Using the chain rule, derivative of D^{-1} is -1 * D^{-2} * derivative of D. The derivative of D with respect to s is -α.

            So putting it all together:

            d/ds φ(s,1) = (1 - α - β) * [ -1 * D^{-2} * (-α) ] = (1 - α - β) * α / D^2.

            So that's equal to α(1 - α - β)/( (1 - β - α s)^2 ).

            Now, evaluate this at s=1. So substitute s=1:

            E(X) = α(1 - α - β)/( (1 - β - α *1)^2 ) = α(1 - α - β)/( (1 - β - α)^2 ).

            Wait, but 1 - β - α is (1 - α - β), which is the same as the numerator. So E(X) becomes α(1 - α - β)/( (1 - α - β)^2 ) = α / (1 - α - β).

            Wait, but hold on. If the denominator is (1 - β - α s)^2, and when s=1, it's (1 - β - α)^2. So the numerator is α(1 - α - β). So yes, simplifying gives α / (1 - α - β). But let's check if that makes sense.

            Wait, but let me verify the steps again because sometimes when dealing with generating functions, especially joint ones, there might be a different approach. Alternatively, maybe there's a different way to compute the expectation without marginalizing first.

            Alternatively, since φ(s,t) is the joint PGF, the expectation E[X] can be found by taking the partial derivative of φ with respect to s at s=1, t=1. Because for joint PGFs, E[X] is the partial derivative with respect to s evaluated at (1,1). Let me recall that.

            Yes, I think that's correct. For the joint PGF φ(s,t) = E[s^X t^Y], the first partial derivative with respect to s at (1,1) is E[X], and similarly for Y. So maybe I don't need to marginalize first. Let me check that.

            So, computing ∂φ/∂s evaluated at s=1, t=1 should give E[X]. Let me try that approach.

            Given φ(s, t) = (1 - α - β)/(1 - α s - β t). Let's compute the partial derivative with respect to s:

            ∂φ/∂s = [ (1 - α - β) * derivative of denominator^{-1} ].

            So denominator is (1 - α s - β t), so derivative with respect to s is -α. So as before:

            ∂φ/∂s = (1 - α - β) * [ -(-α) / (1 - α s - β t)^2 ] = α(1 - α - β)/(1 - α s - β t)^2.

            Then evaluate this at s=1, t=1:

            E[X] = α(1 - α - β)/(1 - α*1 - β*1)^2 = α(1 - α - β)/(1 - α - β)^2.

            Simplify: α(1 - α - β) divided by (1 - α - β)^2 is α/(1 - α - β).

            So that gives the same result as before, which is reassuring. So regardless of whether I compute the marginal PGF first and take the derivative, or take the partial derivative of the joint PGF, I get the same answer, which is α/(1 - α - β).

            Therefore, the expected value of X is α divided by (1 - α - β).

            But I should check if this makes sense. Let's think of the parameters. For the generating function to be valid, the denominator 1 - α s - β t must not be zero when |s| ≤ 1 and |t| ≤ 1. So, assuming that α and β are positive constants such that α + β < 1, so that the denominator 1 - α - β is positive when s = t =1. That's necessary for convergence. So α and β are between 0 and 1, and their sum is less than 1. So 1 - α - β is positive. So the expectation E[X] = α/(1 - α - β). That seems reasonable.

            If α is 0, then E[X] is 0, which makes sense. If β is 0, then the PGF becomes (1 - α)/(1 - α s), which is a geometric distribution with parameter α, and expectation α/(1 - α), but wait, hold on. Wait, if φ(s) = (1 - α)/(1 - α s), that is similar to a geometric distribution's PGF. Wait, the standard geometric distribution with success probability p has PGF p/(1 - (1 - p)s). So if we have φ(s) = (1 - α)/(1 - α s), comparing to the standard PGF, we can write this as [(1 - α)] / [1 - α s]. Let's set that equal to p/(1 - (1 - p)s). Then p = 1 - α, and 1 - p = α. Therefore, this would imply p = 1 - α, so the expectation of a geometric distribution is (1 - p)/p = α/(1 - α). Wait, but in our case, the expectation here is α/(1 - α - β). So when β =0, then it's α/(1 - α), which matches the expectation for the geometric distribution with parameter p =1 - α. Wait, but if φ(s) = (1 - α)/(1 - α s), then the probabilities are (1 - α) α^k for k =0,1,2,... So that is a geometric distribution starting at 0, with probability of success (1 - α), so the expectation is α/(1 - α), which matches our result when β=0. So that seems correct.

            Therefore, in the case where there's a β component, the expectation increases to α/(1 - α - β). That makes sense because if there's another parameter β, maybe there's some dependency or another variable involved, but the marginal expectation of X would still be driven by α and the total 'mass' subtracted by α and β.

            Thus, after carefully computing the derivative and verifying through two methods, both giving the same result, and checking the edge case where β=0 leading to a known expectation, I believe the answer is E(X) = α/(1 - α - β).
            </think>To find the expected value \(\mathbb{E}(X)\) given the joint probability generating function \(\varphi(s, t) = \frac{1 - \alpha - \beta}{1 - \alpha s - \beta t}\), we can use the property of generating functions that the expected value \(\mathbb{E}(X)\) is given by the partial derivative of \(\varphi(s, t)\) with respect to \(s\) evaluated at \(s = 1\) and \(t = 1\).

            1. **Compute the partial derivative of \(\varphi(s, t)\) with respect to \(s\):**

            \[
            \varphi(s, t) = \frac{1 - \alpha - \beta}{1 - \alpha s - \beta t}
            \]

            Using the quotient rule for differentiation, we have:

            \[
            \frac{\partial \varphi(s, t)}{\partial s} = \frac{(1 - \alpha - \beta) \cdot \frac{\partial}{\partial s} (1 - \alpha s - \beta t) - (1 - \alpha s - \beta t) \cdot \frac{\partial}{\partial s} (1 - \alpha - \beta)}{(1 - \alpha s - \beta t)^2}
            \]

            Since \(\frac{\partial}{\partial s} (1 - \alpha s - \beta t) = -\alpha\) and \(\frac{\partial}{\partial s} (1 - \alpha - \beta) = 0\), we get:

            \[
            \frac{\partial \varphi(s, t)}{\partial s} = \frac{(1 - \alpha - \beta) \cdot (-\alpha)}{(1 - \alpha s - \beta t)^2} = \frac{-\alpha (1 - \alpha - \beta)}{(1 - \alpha s - \beta t)^2}
            \]

            2. **Evaluate the partial derivative at \(s = 1\) and \(t = 1\):**

            \[
            \left. \frac{\partial \varphi(s, t)}{\partial s} \right|_{s=1, t=1} = \frac{-\alpha (1 - \alpha - \beta)}{(1 - \alpha \cdot 1 - \beta \cdot 1)^2} = \frac{-\alpha (1 - \alpha - \beta)}{(1 - \alpha - \beta)^2}
            \]

            Simplifying the expression:

            \[
            \left. \frac{\partial \varphi(s, t)}{\partial s} \right|_{s=1, t=1} = \frac{-\alpha (1 - \alpha - \beta)}{(1 - \alpha - \beta)^2} = \frac{-\alpha}{1 - \alpha - \beta}
            \]

            Since \(\mathbb{E}(X)\) is the positive value obtained from the derivative, we take the absolute value:

            \[
            \mathbb{E}(X) = \frac{\alpha}{1 - \alpha - \beta}
            \]

            Therefore, the expected value of \(X\) is:

            \[
            \boxed{\frac{\alpha}{1 - \alpha - \beta}}
            \]
        """
    print(parse(text))




    
    
if __name__ == "__main__":
    main()
