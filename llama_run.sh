# srun -X -p moss -n1 --ntasks-per-node=1 --gres=gpu:8 --exclusive sh llama_run.sh 

# torchrun --nproc_per_node 1 llama_run.py \
# --ckpt_dir=s3://model_weights/0331/evaluation/exported_llama/finetune7b/12000/ \
# --tokenizer_path=/mnt/petrelfs/share_data/llm_llama/tokenizer.model \
# --tokenizer_type=llama --max_seq_len=2048 --max_batch_size=32 

# torchrun --nproc_per_node 2 llama_run.py \
# --ckpt_dir=s3://model_weights/0331/evaluation/exported_llama/7132v2/11100/ \
# --tokenizer_path='/mnt/petrelfs/share_data/llm_weight/final_model_v6.model'\
# --tokenizer_type='v6' --max_seq_len=2048 --max_batch_size=32 

# torchrun --nproc_per_node 8 llama_run.py \
# --ckpt_dir=s3://model_weights/0331/evaluation/exported_llama/1006/12499/ \
# --tokenizer_path=/mnt/petrelfs/share_data/llm_data/tokenizers/llamav4.model \
# --tokenizer_type=v4 --max_seq_len=2048 --max_batch_size=32 

# torchrun --nproc_per_node 8 llama_run.py \
# --ckpt_dir=s3://model_weights/0331/evaluation/exported_llama/1007/7249/ \
# --tokenizer_path=/mnt/petrelfs/share_data/llm_llama/tokenizer.model \
# --tokenizer_type=llama --max_seq_len=2048 --max_batch_size=32 

echo "Done"