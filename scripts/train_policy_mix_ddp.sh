# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0



fusermount -u /mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/oxe
/mnt/petrelfs/liumingyu/s3mount vla_data /mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/oxe --allow-overwrite --allow-delete --endpoint-url http://10.140.27.254:80   

export http_proxy=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/
export https_proxy=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/
export HTTP_PROXY=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/
export HTTPS_PROXY=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/
DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
num_gpus=${5:-2}  # 默认使用2个GPU，可以通过第5个参数指定
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
# gpu_id=${5}
# echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mUsing ${num_gpus} GPUs for training\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
# export CUDA_VISIBLE_DEVICES=${gpu_id}
# --multi_gpu \
accelerate launch \
    --num_processes=${num_gpus} \
    --multi_gpu \
    --mixed_precision=fp16 \
    train_mix_ddp.py --config-name=${config_name}.yaml \
                     task=${task_name} \
                     hydra.run.dir=${run_dir} \
                     training.debug=$DEBUG \
                     training.seed=${seed} \
                     exp_name=${exp_name} \
                     logging.mode=${wandb_mode} \
                     checkpoint.save_ckpt=${save_ckpt}



                                