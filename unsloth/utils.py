import os
from os.path import dirname


SYSTEM_PROMPT = "You are given a math problem. Please reason step by step, and put your final answer within \\boxed{}."
REASONING_START = "<think>"
REASONING_END = "</think>" 


def get_root_dir():
    root = os.path.abspath('')
    project_name = 'rl-math-reasoning'
    print("Project name set as {}. Make sure it is correct!".format(project_name))
    while root.split('/')[-1] != project_name:
        root = dirname(root)
    return root


def create_chat_template(tokenizer):

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
    
    chat_template = chat_template.replace("'{system_prompt}'", f"'{SYSTEM_PROMPT}'")
    chat_template = chat_template.replace("'{reasoning_start}'", f"'{REASONING_START}'")
    
    tokenizer.chat_template = chat_template

    return tokenizer