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

import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL
from drag_pipeline import DragPipeline

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from .drag_utils import drag_diffusion_update
from .lora_utils import train_lora
from .attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl

import torchvision.transforms.functional as Fu

from .shift_test import shift_matrix,copy_past,paint_past
from .continuous_drag import drag_stretch_multipoint_ratio_interp
torch.set_printoptions(profile="full")
import copy

from .unet_drag.unet_2d_condition import UNet2DConditionModel  # for memory

# -------------- general UI functionality --------------
def clear_all(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=True), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None

def clear_all_gen(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None, None

def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out


def store_img_om_fill(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return mask


def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], gr.Image.update(value=masked_img, interactive=True), mask


def store_img_om(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], gr.Image.update(value=masked_img, interactive=True), mask, gr.Image.update(value=image, interactive=True)

# once user upload an image, the original image is stored in `original_image`
# the same image is displayed in `input_image` for point clicking purpose
def store_img_gen(img):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = np.array(image)
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask

# user click the image to get points, and show the points on the image
def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

# clear all handle/target points
def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []
# ------------------------------------------------------

# ----------- dragging user-input image utils -----------
def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress()):
    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress)
    return "Training LoRA Done!"

def preprocess_image(image,
                     device,
                     dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image




def run_drag(source_image,
             image_with_clicks,
             mask,
             prompt,
             points,
             inversion_strength,
             model_path,
             vae_path,
             lora_path,
             start_step,
             start_layer,
             n_inference_step,
             task_cat,
             fill_mode,
             use_kv_cp="default",
             use_lora_ = "default",
             mask_fill = None,
             testif=0,
             save_dir="./results",
    ):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    
    
    # add for lcm
    use_lcm_unet=True
    use_lcm_lora_sdv15=False
    if use_lcm_unet:
        # print('use lcm unet')
        unet = UNet2DConditionModel.from_pretrained(
                    "SimianLuo/LCM_Dreamshaper_v7",
                    subfolder="unet",
                    torch_dtype=torch.float16,)
        model.unet = unet
    else:
        # print('not use lcm unet')
        unet = UNet2DConditionModel.from_pretrained(
                    model_path,
                    subfolder="unet",
                    torch_dtype=torch.float16,)
        model.unet = unet
    if use_lcm_lora_sdv15:
        # print('use lcm lora sdv1.5')
        model.load_lora_weights("lcm-lora-sdv1-5")

    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    # off load model to cpu, which save some memory.
    model.enable_model_cpu_offload()

    # initialize parameters
    seed = 42 # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    # print("prompt:", prompt)
    args.prompt = prompt
    args.points = points
    args.n_inference_step = int(n_inference_step) #50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0

    args.unet_feature_idx = [3]

    args.r_m = 1
    args.r_p = 3
    # args.lam = lam

    # args.lr = latent_lr
    # args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    print(args)

    source_image = preprocess_image(source_image, device, dtype=torch.float16)
    if testif == 0:
        image_with_clicks = preprocess_image(image_with_clicks, device)

    # preparing editing meta data (handle, target, mask)
    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")    # 1: editing(masked), 0: not editing

    if mask_fill is None:
        pass
    else:
        print(f"mask fill 1: {mask_fill.shape}")
        mask_fill = torch.from_numpy(mask_fill).float() / 255.
        mask_fill[mask_fill > 0.0] = 1.0
        mask_fill = rearrange(mask_fill, "h w -> 1 1 h w").cuda()
        mask_fill = F.interpolate(mask_fill, (args.sup_res_h, args.sup_res_w), mode="nearest")
        mask_fill = Fu.resize(mask_fill, (int(mask_fill.shape[-2]/4), int(mask_fill.shape[-1]/4))) # some image are not h==w

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)

    # add
    shift_yx = target_points[0]-handle_points[0] # only one point
    shift_yx = shift_yx.to(device=source_image.device)

    # mask_cp_handle = Fu.resize(mask, (64, 64))
    mask_cp_handle = Fu.resize(mask, (int(mask.shape[-2]/4), int(mask.shape[-1]/4))) # some image are not h==w
    shift_y,shift_x= int(shift_yx[0]/4),int(shift_yx[1]/4)
    mask_cp_target = shift_matrix(mask_cp_handle, shift_x, shift_y)

    # set lora (optimal)
    use_dragdiff = 0
    use_noise_copy = 0

    use_lora = 0
    use_rlc_attn = 0
    use_drag_stretch = 0

    use_kv_copy= 1
    use_onestep_latent_copy = 1     # copy in one step, only in last latent
    use_substep_latent_copy = 0  

    if task_cat == "object moving":
        tex = f"_om"
        print("the task is object moving")
        use_lora = 0
        use_rlc_attn = 0
        use_drag_stretch = 0
        use_kv_copy= 1
        use_onestep_latent_copy = 1
    elif task_cat == "continuous drag":
        tex = f"_cd"
        print("the task is continuous drag")
        use_lora = 0
        use_rlc_attn = 0
        use_drag_stretch = 1
        use_kv_copy= 1
        use_onestep_latent_copy = 0
    else:
        tex = f"_om"
        print(f"warning:no this task '{task_cat}' \nset the task is object moving")

    if use_kv_cp == "not use":
        use_kv_copy = 0
    elif use_kv_cp == "use":
        use_kv_copy = 1
    if use_kv_copy:
        pass
        # print("use kv copy")
    else:
        pass
        # print("not use kv copy")

    if use_lora_ == "not use":
        use_lora = 0
        use_rlc_attn = 0
    elif use_lora_ == "use":
        use_lora = 1
        use_rlc_attn = 1

    if not use_lora:
        pass
        # print("do not use lora")
    elif lora_path == "":
        # print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        # print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # obtain text embeddings
    text_embeddings = model.get_text_embeddings(prompt)
    # if text_embeddings is None:
    #     print("text_embeddings is none")

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(source_image,
                               prompt,
                               text_embeddings=text_embeddings,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step,
                               mask_cp_target=mask_cp_target,
                               shift_yx=shift_yx,
                               use_noise_copy=use_noise_copy,
                               mask_cp_handle = mask_cp_handle,
                               handle_point = torch.round(handle_points[0]/4).to(device=mask_cp_handle.device),
                               target_point = torch.round(target_points[0]/4).to(device=mask_cp_handle.device),
                               use_substep_latent_copy = use_substep_latent_copy,
                               use_kv_copy=use_kv_copy
                               )

    if use_onestep_latent_copy:
        # print("******use copy paste*******")
        invert_code_d = copy.deepcopy(invert_code)
        invert_code = copy_past(invert_code,mask_cp_target,shift_yx)
        invert_code = paint_past(invert_code,invert_code_d,mask_cp_handle,mask_cp_target,target_points[0]/4,mask_fill=mask_fill)

    if use_drag_stretch:
        # print("******use drag stretch*******")
        invert_code = drag_stretch_multipoint_ratio_interp(invert_code=invert_code,
                                    handle_points=handle_points,
                                    target_points=target_points,
                                    mask_cp_handle=mask_cp_handle,
                                    fill_mode=fill_mode)
        
    # empty cache to save memory
    torch.cuda.empty_cache()

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    init_code = init_code.float()
    text_embeddings = text_embeddings.float()
    model.unet = model.unet.float()
    if use_dragdiff:
        updated_init_code = drag_diffusion_update(model, init_code,
            text_embeddings, t, handle_points, target_points, mask, args,
            use_kv_copy=use_kv_copy)
        updated_init_code = updated_init_code.half()
    text_embeddings = text_embeddings.half()
    model.unet = model.unet.half()

    # empty cache to save memory
    torch.cuda.empty_cache()

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=start_step,
                                        start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    
    # it is to do the dragdiffusion Reference-Latent-Control (rlc)
    # but it is not need in fastdrag
    if not use_rlc_attn:
        pass
        # print('do not use rlc')
    elif lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    # add use_noise_copy
    # inference the synthesized image
    if use_noise_copy or use_onestep_latent_copy or use_substep_latent_copy or use_drag_stretch:
        # print('sample from latent directly')
        gen_image = model(
            prompt=args.prompt,
            text_embeddings=torch.cat([text_embeddings, text_embeddings], dim=0),
            batch_size=2,
            latents=torch.cat([invert_code, invert_code], dim=0),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step
            )[1].unsqueeze(dim=0)
    else:
        gen_image = model(
            prompt=args.prompt,
            text_embeddings=torch.cat([text_embeddings, text_embeddings], dim=0),
            batch_size=2,
            latents=torch.cat([init_code_orig, updated_init_code], dim=0),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step
            )[1].unsqueeze(dim=0)
    
    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')
    if testif == 0:
        # save the original image, user editing instructions, synthesized image
        save_result = torch.cat([
            source_image.float() * 0.5 + 0.5,
            torch.ones((1,3,full_h,25)).cuda(),
            image_with_clicks.float() * 0.5 + 0.5,
            torch.ones((1,3,full_h,25)).cuda(),
            gen_image[0:1].float()
        ], dim=-1)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

        # save img splitly (optimal)
        # tex += f"_n{args.n_inference_step}"
        # save_dir_s = os.path.join(save_dir,save_prefix+tex)
        # save_dir_s_ori = os.path.join(save_dir_s,'ori')
        # save_dir_s_cli = os.path.join(save_dir_s,'cli')
        # save_dir_s_gen = os.path.join(save_dir_s,'gen')
        # os.makedirs(save_dir_s,exist_ok=True)
        # os.makedirs(save_dir_s_ori,exist_ok=True)
        # os.makedirs(save_dir_s_cli,exist_ok=True)
        # os.makedirs(save_dir_s_gen,exist_ok=True)
        # save_image(save_result, os.path.join(save_dir_s, save_prefix + '.png'))
        # save_image(source_image.float() * 0.5 + 0.5, os.path.join(save_dir_s_ori, save_prefix + '_ori.png'))
        # save_image(image_with_clicks.float() * 0.5 + 0.5, os.path.join(save_dir_s_cli, save_prefix + '_cli.png'))
        # save_image(gen_image[0:1].float(), os.path.join(save_dir_s_gen, save_prefix + '_gen.png'))
        # print(f"save finsh!:{os.path.join(save_dir_s, save_prefix + '.png')}")

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return gr.Image.update(value=out_image)

# -------------------------------------------------------

