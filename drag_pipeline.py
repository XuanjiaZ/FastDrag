# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) FastDrag Authors 
#
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

import torch
import numpy as np
import os

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers import StableDiffusionPipeline

from utils.shift_test import shift_matrix

import copy
from utils.shift_test import shift_matrix,copy_past,paint_past,get_mask_of_point,get_complementary_of_mask
from utils.utils import split_into_N_equal_parts


torch.set_printoptions(profile="full")
# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks
def override_forward(self):

    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        last_up_block_idx: int = None,
        iter_cur=0, save_kv=True
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        # print('sample.dtype 0 ',sample.dtype)
        sample = self.conv_in(sample)
        # print('sample.dtype 0 ',sample.dtype)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    iter_cur=iter_cur, save_kv=save_kv
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                iter_cur=iter_cur, save_kv=save_kv
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        # the 0-th element is the mid-block output
        all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    iter_cur=iter_cur, save_kv=save_kv
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            all_intermediate_features.append(sample)
            # return early to save computation time if needed
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results
        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward


class DragPipeline(StableDiffusionPipeline):

    # must call this function when initialize
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        """
        predict the sample of the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        # print('image shape2 ', image.shape) # torch.Size([1, 3, 512, 512])
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        # print('latents shape ', latents.shape)  # torch.Size([1, 4, 64, 64])
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        return text_embeddings

    # get all intermediate features and then do bilinear interpolation
    # return features in the layer_idx list
    def forward_unet_features(self, z, t, encoder_hidden_states, layer_idx=[0], interp_res_h=256, interp_res_w=256,
                              use_kv_copy=1):
        save_kv=use_kv_copy
        unet_output, all_intermediate_features = self.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True,
            save_kv=save_kv
            )

        all_return_features = []
        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)
        return unet_output, return_features

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        text_embeddings=None, # whether text embedding is directly provided.
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        latents=None,
        neg_prompt=None,
        return_intermediates=False,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if text_embeddings is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size
            # text embeddings
            text_embeddings = self.get_text_embeddings(prompt)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_embeddings = self.get_text_embeddings([uc_text]*batch_size)
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        if return_intermediates:
            latents_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            # print("sampling t input:", t)
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings, iter_cur=i, save_kv=False)
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            # YUJUN: right now, the only difference between step here and step in scheduler
            # is that scheduler version would clamp pred_x0 between [-1,1]
            # don't know if that's gonna have huge impact
            # print("latents shape & dtype: 0", latents.shape, latents.dtype)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents=latents.to(dtype=torch.float16)
            # print("latents shape & dtype: 1", latents.shape, latents.dtype)
            if return_intermediates:
                latents_list.append(latents)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        text_embeddings=None,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        mask_cp_target=None,
        shift_yx=None,
        use_noise_copy=None,
        mask_cp_handle=None,
        handle_point = None,
        target_point = None,
        use_substep_latent_copy = True,
        use_kv_copy = 1,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if text_embeddings is None:
            if isinstance(prompt, list):
                if batch_size == 1:
                    image = image.expand(len(prompt), -1, -1, -1)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size
            text_embeddings = self.get_text_embeddings(prompt)

        # define initial latents
        latents = self.image2latent(image)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]

        shift_yx = torch.round(shift_yx/4)
        if use_substep_latent_copy:
            sub_steps =  num_actual_inference_steps-4 #num_actual_inference_steps
            sub_shiftys = split_into_N_equal_parts(shift_yx[0],sub_steps)
            sub_shiftxs = split_into_N_equal_parts(shift_yx[1],sub_steps)
            sub_shift_yxs = torch.stack([sub_shiftys,sub_shiftxs],dim=1)
            sub_flag = 0


        save_kv = use_kv_copy
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            # print("the input t:", t)
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,iter_cur=len(self.scheduler.timesteps)-i-1, save_kv=save_kv) # change this for memory
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)


            if use_noise_copy:
                # print('***'*20,f'noise befroe\n{noise_pred}')
                ndtype = noise_pred.dtype
                noise_pred_d = copy.deepcopy(noise_pred)
                # print('noise_pred 0 ',noise_pred.shape)
                shift_y,shift_x = int(shift_yx[0]),int(shift_yx[1])
                noise_cp_target = shift_matrix(noise_pred_d,shift_x,shift_y)*mask_cp_target
                noise_pred = noise_pred*(1-mask_cp_target)+noise_cp_target
                noise_pred = noise_pred.to(dtype=ndtype)
                # print('***'*20,f'noise after\n{noise_pred}')

                ldtype = latents.dtype
                latents_d = copy.deepcopy(latents)
                latents_cp_target = shift_matrix(latents_d,shift_x,shift_y)*mask_cp_target
                latents = latents*(1-mask_cp_target) + latents_cp_target
                latents = latents.to(dtype=ldtype)

            # print('noise_pred 1 ',noise_pred.shape)


            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            if use_substep_latent_copy and i>=num_actual_inference_steps-int(sub_shift_yxs.shape[0]):
                # need para: mask_cp_handle  mask_cp_target  shift_yx
                print(f'sub_shift_yxs:{sub_shift_yxs} \nsub_flag:{sub_flag} \nsub_shift_yxs[:sub_flag+1].sum(dim=0){sub_shift_yxs[:sub_flag+1].sum(dim=0)}')
                sub_shift_yx = sub_shift_yxs[sub_flag]
                # sub_shift_yx = torch.round(shift_yx/3)   # should be optimized

                img_scale = 0.3
                noise_scale = (1 - img_scale ** 2) ** (0.5)
                # paint = img_scale*latents + noise_scale*torch.randn_like(latents)
                paint = copy.deepcopy(latents)
                # paint = torch.zeros_like(latents) + torch.mean(latents) # mean

                mask_cp_target = get_mask_of_point(template = mask_cp_target, point = handle_point+sub_shift_yxs[:sub_flag+1].sum(dim=0), flag = f'tar_{i}')
                mask_cp_handle = get_mask_of_point(template = mask_cp_handle, point = handle_point+sub_shift_yxs[:sub_flag].sum(dim=0), flag = f'han_{i}')
                mask_cp_handle = get_complementary_of_mask(mask_target=mask_cp_target, mask_handle=mask_cp_handle,flag = i)
                latents = copy_past(latents,mask_cp_target,sub_shift_yx)
                # latents = paint_past(latents,paint,mask_cp_handle)
                latents = paint_past(latents,paint,mask_cp_handle,target_point)
                print(f"sub step copy i:{i} \nsub_shift_yx:{sub_shift_yx} {handle_point+sub_shift_yx*(i-num_actual_inference_steps+3)}\n")
                
                sub_flag +=1
            

            latents_list.append(latents)
            pred_x0_list.append(pred_x0)


        # self.visual_latent(latents)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents
    

    # add for visual
    def visual_latent(self, latent,label=""):
        '''
        latents shape  torch.Size([1, 4, 64, 64])
        '''
        savepath = "./latent_visual"
        os.makedirs(savepath,exist_ok=True)
        img_save_path = os.path.join(savepath,f"latent_{label}.png")

        img = self.latent2image(latent)
        print(f"latent2image img shape:{img.shape}")    # latent2image img shape:(512, 512, 3)
        img_save = Image.fromarray(img)
        print(f"latent2image img_save shape:{img.shape}") 
        img_save.save(img_save_path)
        return img