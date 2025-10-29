def create_chat_template(tokenizer):
    reasoning_start = "<think>"
    reasoning_end   = "</think>"  
    system_prompt = "You are given a math problem. Please reason step by step, and put your final answer within \\boxed{}."

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
    
    chat_template = chat_template.replace("'{system_prompt}'", f"'{system_prompt}'")
    chat_template = chat_template.replace("'{reasoning_start}'", f"'{reasoning_start}'")
    
    tokenizer.chat_template = chat_template

    return tokenizer, system_prompt