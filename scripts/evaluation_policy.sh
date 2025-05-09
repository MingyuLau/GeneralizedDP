# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0
#<lxy>

DEBUG=False

alg_name=${1}
task_name=${2}
test_task_name=${3}
config_name=${alg_name}
addition_info=${4}
seed=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=${6}
#首先获得当前任务的初始参考轨迹
cd third_party/Metaworld
task_name=${task_name}
python get_initial_trajectory.py --env_name=${test_task_name} \
            --root_dir "../../3D-Diffusion-Policy/test_data/" \
            --test_traj_num 10 \
            --seed_base ${seed} \
            


cd ../../3D-Diffusion-Policy
# echo 当前目录：$(pwd)
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python evaluation.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            test_task_name=${test_task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}


#</lxy>
                                