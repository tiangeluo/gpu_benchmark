# gpu_benchmark

### ollama model
Please first install [ollama](https://ollama.com/download/linux): `curl -fsSL https://ollama.com/install.sh | sh`. Then, download the target model ckpt. For example, run `ollama run llama3.1:405b-text-q4_K_M` to download 405B-Q4-KM and then use the script `benchmark_llama31_xxx_ollama.py` to test.

### 405B Q4
Please first download model from https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 and then run `benchmark_llama31_405Q4.py`. You need to install `torch transformers autoawq accelerate` based on your CUDA version.

---

script for running inference testing:
```
torchrun --nproc_per_node 1 benchmark_llama3_inference.py \
    --ckpt_dir /home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B \
    --tokenizer_path /home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4

torchrun --nproc_per_node 8 benchmark_token_s.py \
    --ckpt_dir /home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-70B \
    --tokenizer_path /home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-70B/tokenizer.model \
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

command for full-finetune Llama-3-8B models with single and 4 GPUs.

```bash
tune run full_finetune_single_device --config ./torchtune_configs/8B_full_single_device_wandb.yaml checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model' checkpointer.output_dir='./test' batch_size=8

tune run --nproc_per_node 4 full_finetune_distributed --config ./torchtune_configs/8B_full_wandb.yaml  checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model' checkpointer.output_dir='./test' batch_size=8

tune run --nproc_per_node 4 full_finetune_distributed --config ./torchtune_configs/8B_full_wandb_bs16.yaml  checkpointer.checkpoint_dir='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B' tokenizer.path='/home/tiangel/turbo/shared_datasets/Llama_weights/Meta-Llama-3-8B/tokenizer.model' checkpointer.output_dir='./test' batch_size=16
```
