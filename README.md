# gpu_benchmark

```
# script for running inference testing
torchrun --nproc_per_node 1 benchmark_llama3_inference.py \
    --ckpt_dir /home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B \
    --tokenizer_path /home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

For finetuning, lora, and Q-lora performance, I used `torchtune`, https://github.com/pytorch/torchtune/tree/main. I list some commands below where you should can just change model_dir.

```
tune run lora_finetune_single_device --config llama3/8B_lora_single_device checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model' checkpointer.output_dir='./test' batch_size=16

tune run lora_finetune_single_device --config llama3/8B_qlora_single_device checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model' checkpointer.output_dir='./test' batch_size=1

tune run full_finetune_single_device --config llama3/8B_full_single_device checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model' checkpointer.output_dir='./test' batch_size=8


tune run --nproc_per_node 4 full_finetune_distributed --config llama3/8B_full checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model' checkpointer.output_dir='./test' batch_size=16


tune run --nproc_per_node 8 lora_finetune_distributed --config ./70B_lora.yaml checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-70b' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-70b/tokenizer.model' checkpointer.output_dir='./test' batch_siz
e=8

tune download meta-llama/Meta-Llama-3-70b --hf-token xxx --output-dir ./Meta-Llama-3-70b --ignore-patterns "original/consolidated*"
```


