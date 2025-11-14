python sft.py model_name=unsloth/Qwen3-0.6B

python rl.py lora_config=/home/ujan/rl-math-unsloth/models/sft_checkpoints/checkpoint-684

python rl.py model_name=unsloth/Llama-3.2-1B-Instruct max_steps=2000 save_steps=500