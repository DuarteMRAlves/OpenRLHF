#!/bin/bash

#SBATCH --job-name=gsm8k-test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=3
#SBATCH --partition=a6000
#SBATCH --qos=gpu-medium

module load python/3.12.5
module load cuda/12.4.1

source /mnt/data-artemis/duarte/openrlhf-venv/bin/activate

# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${SLURM_GPUS_PER_TASK}

run_dir=/mnt/scratch-artemis/duarte/reasoning/runs/test-gsm8k
ckpt_dir=${run_dir}/checkpoints
tb_dir=${run_dir}/tensorboard

ray job submit --address="http://127.0.0.1:8265" \
    -- python3 -u -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain Qwen/Qwen2.5-1.5B-Instruct \
    --save_path ${run_dir} \
    --save_steps 100 \
    --logging_steps 1 \
    --eval_steps -1 \
    --num_episodes 2 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 16 \
    --max_epochs 1 \
    --prompt_max_len 512 \
    --generate_max_len 512 \
    --zero_stage 1 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /mnt/scratch-artemis/duarte/reasoning/gsm8k_dataset \
    --input_key messages \
    --apply_chat_template \
    --max_samples 100000000 \
    --normalize_reward \
    --flash_attn \
    --remote_rm_url /mnt/home/duarte/OpenRLHF/examples/gsm8k/reward_function.py \
    --advantage_estimator reinforce_baseline \
    --n_samples_per_prompt 64 \
    --enable_prefix_caching \
    --packing_samples \
    --liger_kernel \
    --use_tensorboard ${tb_dir} \
    --ckpt_path ${ckpt_dir} \
    --load_checkpoint
