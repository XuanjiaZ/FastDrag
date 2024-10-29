# *************************************************************************
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

# run results of FastDrag
import argparse
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import PIL
from PIL import Image

from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

from diffusers import DDIMScheduler, AutoencoderKL
from torchvision.utils import save_image
from pytorch_lightning import seed_everything

import sys
sys.path.insert(0, '../')
from drag_pipeline import DragPipeline

from utils.attn_utils import MutualSelfAttentionControl

from utils.ui_utils import run_drag
import time

def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--lora_steps', type=int, default=80, help='number of lora fine-tuning steps')
    parser.add_argument('--inv_strength', type=float, default=0.7, help='inversion strength')
    parser.add_argument('--latent_lr', type=float, default=0.01, help='latent learning rate')
    parser.add_argument('--unet_feature_idx', type=int, default=3, help='feature idx of unet features')
    parser.add_argument('--result_dir', type=str, default=None, help='feature idx of unet features')
    parser.add_argument('--n_inference_step', type=int, default=10, help='feature idx of unet features')
    args = parser.parse_args()

    all_category = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]

    # assume root_dir and lora_dir are valid directory
    root_dir = 'path to DragBench'
    lora_dir = 'path to drag_bench_lora'
    if args.result_dir == None:
        result_dir = 'drag_diffusion_0506_inter_nolora_kvcopy_inverse10_' + \
            '_' + str(args.lora_steps) + \
            '_' + str(args.inv_strength) + \
            '_' + str(args.latent_lr) + \
            '_' + str(args.unet_feature_idx)
    else:
        result_dir = args.result_dir+ \
            '_' + str(args.lora_steps) + \
            '_' + str(args.inv_strength) + \
            '_' + str(args.latent_lr) + \
            '_' + str(args.unet_feature_idx)

    # mkdir if necessary
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        for cat in all_category:
            os.mkdir(os.path.join(result_dir,cat))
    save_time_sum = 0
    start_time = time.time()
    for cat in all_category:
        file_dir = os.path.join(root_dir, cat)
        for sample_name in os.listdir(file_dir):
            if sample_name == '.DS_Store':
                continue
            sample_path = os.path.join(file_dir, sample_name)

            # read image file
            source_image = Image.open(os.path.join(sample_path, 'original_image.png'))
            source_image = np.array(source_image)

            # load meta data
            with open(os.path.join(sample_path, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']
            mask = meta_data['mask']
            points = meta_data['points']

            # load lora
            lora_path = os.path.join(lora_dir, cat, sample_name, str(args.lora_steps))
            print("applying lora: " + lora_path)

            image_with_clicks = None
            out_image = run_drag(source_image,
                                image_with_clicks,
                                mask,
                                prompt,
                                points,
                                inversion_strength = args.inv_strength,
                                model_path='path to stable-diffusion-v1-5',
                                vae_path="default",
                                lora_path=lora_path, # is not used 
                                start_step=0,
                                start_layer=10,
                                n_inference_step=args.n_inference_step,
                                task_cat="continuous drag",
                                fill_mode='interpolation', 
                                use_kv_cp="default",
                                use_lora_ = "default",
                                testif=1,
                                save_dir="./results",)
            end_time = time.time()
            save_time0 = time.time()
            save_dir = os.path.join(result_dir, cat, sample_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            Image.fromarray(out_image).save(os.path.join(save_dir, 'dragged_image.png'))
            save_time1 = time.time()
            save_time_sum += (save_time1-save_time0)
    print(f"***************\n"*2)
    print(f"use time sum: {end_time-start_time}")
    print(f"use save time sum: {save_time_sum}")
    print(f"use drag time sum: {end_time-start_time-save_time_sum}")
    print(f"use drag time per point: {(end_time-start_time-save_time_sum)/349}")
    print(f"***************\n"*2)
    logg = f"***************\n"*2 + \
            f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())}\n" +\
            f"{result_dir}:  \n" + \
            f"use time sum: {end_time-start_time} \n" + \
            f"use save time sum: {save_time_sum} \n" + \
            f"use drag time sum: {end_time-start_time-save_time_sum} \n" +\
            f"use drag time per point: {(end_time-start_time-save_time_sum)/349}\n\n\n"
    with open("./run_drag_result.txt", 'a') as f:
        f.write(logg)