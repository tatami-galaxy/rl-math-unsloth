#### trl sft
CUDA_VISIBLE_DEVICES=0,1 python trl_sft.py --model_name Qwen/Qwen3-4B-Base --model_revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 --max_seq_len 16384 --per_device_train_batch_size 2 --gradient_accumulation_steps 16

CUDA_VISIBLE_DEVICES=0,1,2,3 python trl_sft.py --model_name Qwen/Qwen3-4B-Base --model_revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 --max_seq_len 16384 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --pad_to_multiple_of 8

#### unsloth sft
python sft.py model_name=unsloth/Qwen3-0.6B
python sft.py model_name=unsloth/Qwen3-4B-Base full_finetuning=True
python sft.py model_name=unsloth/Qwen3-4B-Base load_in_16bit=True

#### unsloth rl
python rl.py lora_config=/home/ujan/rl-math-unsloth/models/sft_checkpoints/checkpoint-684
python rl.py model_name=unsloth/Llama-3.2-1B-Instruct max_steps=2000 save_steps=500