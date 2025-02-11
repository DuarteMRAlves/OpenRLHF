#!/bin/bash

#SBATCH --job-name=count-down-test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --partition=a6000
#SBATCH --qos=gpu-debug

module load python/3.12.5
module load cuda/12.4.1

source /mnt/data-artemis/duarte/openrlhf-venv/bin/activate

mkdir countdown-test/

# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${SLURM_GPUS_PER_TASK}

ray job submit --address="http://127.0.0.1:8265" \
    -- python3 -u -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain Qwen/Qwen2.5-1.5B-Instruct \
    --save_path /mnt/scratch-artemis/duarte/reasoning/runs/test-countdown \
    --save_steps 100 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 1 \
    --train_batch_size 4 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 0 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /mnt/scratch-artemis/duarte/reasoning/countdown_dataset \
    --input_key messages \
    --apply_chat_template \
    --max_samples 100000000 \
    --normalize_reward \
    --flash_attn \
    --remote_rm_url /mnt/home/duarte/OpenRLHF/examples/contdown/reward_function.py \
    --advantage_estimator reinforce \
    --packing_samples
