from dataclasses import dataclass, field
import random

import csv
from math_verify import parse, verify
from tqdm.auto import tqdm
from datasets import load_dataset

from transformers import HfArgumentParser
from transformers.utils import logging
logging.set_verbosity_error()


def create_prompt(model_args, question):
    think = model_args.think
    no_think = model_args.no_think
    prompt_type = model_args.prompt_type
    # prompt type
    if prompt_type == 'few_shot':
        raise NotImplementedError
    elif prompt_type == 'zero_shot':
        prompt = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion : {}\n\nAnswer :"
    # add think token to enable thinking
    if think:
        prompt = prompt + '/think'
    # disable thinking
    elif no_think:
        prompt = prompt + '/no_think'
    # final prompt
    formated_prompt = prompt.format(question)
    final_prompt = [{"role": "user", "content": formated_prompt}]
    return final_prompt


def run_model(vllm, prompt):
    output = vllm.query_vllm_server(prompt)
    # remove think tokens from output
    if '</think>' in output:
        output = output.split('</think>')[-1].strip('\n')
    return output


def get_accuracy(gold, answer):
    if verify(parse(gold), parse(answer)):
        return 1.0
    else:
        return 0.0



@dataclass
class DataArguments:

    data_dir: str = field(default=None)
    sample: bool = field(default=False)
    num_samples: int = field(default=100)
    seed: int = field(default=42)


@dataclass
class ModelArguments:

    model_name: str = field(default=None)
    think : bool = field(default=False)
    no_think : bool = field(default=False)
    port: int = field(default=None)
    prompt_type: str = field(default=None)



if __name__ == "__main__":

    # parse cl arguments
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()

    # set seed
    random.seed(data_args.seed)

    # get root dir
    root = get_root_dir()

    # data directory to save info
    if data_args.data_dir is None:
        data_dir = root + '/data/interim/'
    else:
        data_dir = data_args.data_dir

    # load dataset
    dataset_name = 'HuggingFaceH4/MATH-500'
    dataset = load_dataset(dataset_name)['test']

    # sample dataset
    if data_args.sample:
        dataset = dataset.select(
            random.sample(list(range(len(dataset))), data_args.num_samples)
        )

    # vllm object
    vllm = VLLM(model_args.model_name, model_args.port)

    # eval
    bar = tqdm(range(len(dataset)))
    accuracy = 0.0
    incorr = []
    for example in dataset:
        
        question = example['problem']
        answer = example['solution']

        # prompt
        prompt = create_prompt(model_args, question)

        # query model
        output = run_model(vllm, prompt)

        # get accuracy
        acc = get_accuracy(answer, output)
        accuracy += acc

        # store info
        if acc == 0.0:
            incorr.append((question, answer, output))

        bar.update(1)

    print(accuracy/len(dataset))
