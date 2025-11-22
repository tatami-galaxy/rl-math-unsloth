#### trl sft
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file /home/ujan/.cache/huggingface/accelerate/cp_
4_config.yaml trl_sft.py --model_name Qwen/Qwen3-4B-Base --model_revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 --max_seq_len 16384 --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --pad_to_multiple_of 8

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file /home/ujan/.cache/huggingface/accelerate/cp_2_config.yaml trl_sft_with_eval.py --model_name Qwen/Qwen3-4B-Base --model_revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 --max_seq_len 8192 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --pad_to_multiple_of 4

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file /home/ujan/.cache/hugg
ingface/accelerate/cp_2_config.yaml trl_sft_with_eval.py --model_name Qwen/Qwen3-4B-Base --model_revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 --max_seq_len 4096 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --pad_to_multiple_of 4 --eval_steps 10 --looging_steps 10 --save_steps 10

#### unsloth sft
python sft.py model_name=unsloth/Qwen3-0.6B
python sft.py model_name=unsloth/Qwen3-4B-Base full_finetuning=True
python sft.py model_name=unsloth/Qwen3-4B-Base load_in_16bit=True

#### unsloth rl
python rl.py lora_config=/home/ujan/rl-math-unsloth/models/sft_checkpoints/checkpoint-684
python rl.py model_name=unsloth/Llama-3.2-1B-Instruct max_steps=2000 save_steps=500