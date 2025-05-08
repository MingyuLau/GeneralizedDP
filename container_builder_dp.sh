export http_proxy=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/;export https_proxy=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/;export HTTP_PROXY=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/;export HTTPS_PROXY=http://liumingyu:lkUAl4PtbY9KNbRuzZT2Oq0DxkVpnhscuohj3wJNOAK9woBmZygKnE35omts@10.1.20.50:23128/

export APPTAINER_TMPDIR=/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/apptainer-dp3

export ENV_NAME=dp3

apptainer build --bind /mnt:/mnt ${ENV_NAME}.sif  ${ENV_NAME}.def
