CONDA_ROOT=$HOME/anaconda3
ENV_NAME=pytorch3d

GCC_ROOT=/mnt/petrelfs/share/gcc/gcc-11.2.0
MPC_ROOT=/mnt/petrelfs/share/gcc/mpc-0.8.1
MPFR_ROOT=/mnt/petrelfs/share/gcc/mpfr-4.1.0
GMP_ROOT=/mnt/petrelfs/share/gcc/gmp-6.2.0

CUDA_ROOT=/mnt/petrelfs/share/cuda-12.1


# cuda
export PATH=${CUDA_ROOT}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=${CUDA_ROOT}

# gcc
export PATH=${GCC_ROOT}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${GCC_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${MPC_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${MPFR_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${GMP_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# conda
export PATH=${CONDA_ROOT}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CONDA_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# apptainer
export APPTAINER_CACHEDIR="/tmp"

# python user base
export PYTHONUSERBASE=${HOME}/.local/${ENV_NAME}
export PATH=${PYTHONUSERBASE}/bin${PATH:+:${PATH}}


source activate
conda deactivate
conda activate ${CONDA_ROOT}/envs/${ENV_NAME}

export LD_PRELOAD=${GCC_ROOT}/lib64/libstdc++.so.6:/mnt/lustre/share/glibc-2.27/lib/libm-2.27.so