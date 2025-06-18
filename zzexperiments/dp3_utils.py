"""Utils for evaluating the OpenVLA policy."""

import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, ConcatDataset
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
sys.path.append("/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/3D-Diffusion-Policy/")
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from omegaconf import OmegaConf
import hydra
import pathlib
import dill
# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# dp3_cfg = OmegaConf.load("/mnt/petrelfs/liumingyu/code/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/libero_calvin_maniskill-dp3-all_0_seed0/.hydra/config.yaml")

def get_dp3(cfg):
    """Loads and returns a DP3 model from checkpoint."""
    # Load DP3 checkpoint.
    print("[*] Instantiating Pretrained DP3 model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")
    # dp3_cfg = OmegaConf.load("/home/hz/Downloads/config.yaml")
    # model = hydra.utils.instantiate(dp3_cfg.policy)
    # lastest_ckpt_path = "/home/hz/Downloads/epoch=0024-test_mean_score=-0.004.ckpt"
    # path = pathlib.Path(lastest_ckpt_path)
    
    dp3_cfg = OmegaConf.load("/home/hz/Downloads/nopcd_config.yaml")
    model = hydra.utils.instantiate(dp3_cfg.policy)
    # lastest_ckpt_path = "/home/hz/Downloads/latest.ckpt"
    # import pdb; pdb.set_trace()
    # lastest_ckpt_path = "/home/hz/Downloads/nopcd_epoch=0029-test_mean_score=-0.004.ckpt"
    # lastest_ckpt_path = "/home/hz/Downloads/nopcd_epoch=0586-test_mean_score=-0.000.ckpt"
    lastest_ckpt_path = "/home/hz/Downloads/lmy_debug_latest.ckpt"
    path = pathlib.Path(lastest_ckpt_path)
    
    payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
    model.load_state_dict(payload['state_dicts']['model'])

    return model


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def get_dp3_action(model, obs):
    """Generates an action with the DP3 policy."""


    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    # import pdb; pdb.set_trace()
    result = model.predict_action(obs)
    return result

        # Convert back to PIL Image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=True, top_p=0.75)
    return action


def get_vla_latent_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, hist_action=''):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    
    if len(hist_action) > 0:
        prompt = f"In: What action should the robot take to {task_label.lower()}? History action {hist_action}\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(vla.device, dtype=torch.bfloat16)

    # Get latent action.
    action = vla.predict_latent_action(**inputs, unnorm_key=unnorm_key, do_sample=True, temperature=0.75, top_p = 0.9)

    return action