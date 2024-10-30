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
import gradio as gr

from utils.ui_utils import get_points, undo_points
from utils.ui_utils import clear_all, store_img, store_img_om, train_lora_interface, run_drag, store_img_om_fill

LENGTH=480 # length of the square area displaying/editing images
LENGTH_OM = 720
local_models_dir = "./local_pretrained_models"

with gr.Blocks() as demo:
    # layout definition
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of [FastDrag]
        """)

    # UI components for editing real images
    with gr.Tab(label="Continuous Drag"):
        mask = gr.State(value=None) # store mask
        selected_points = gr.State([]) # store points
        original_image = gr.State(value=None) # store original input image
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                    show_label=True, height=LENGTH, width=LENGTH) # for mask painting
                train_lora_button = gr.Button("Train LoRA (optional)")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                input_image = gr.Image(type="numpy", label="Click Points",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False) # for points clicking
                undo_button = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
                output_image = gr.Image(type="numpy", label="Editing Results",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)
                with gr.Row():
                    run_button = gr.Button("Run")
                    clear_all_button = gr.Button("Clear All")

        # general parameters
        with gr.Row():
            prompt = gr.Textbox(label="Prompt (optional)")
            lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path (optional)")
            lora_status_bar = gr.Textbox(label="display LoRA training status")
            n_inference_step = gr.Textbox(value="10", label="n_inference_step") # add
            task_cat = gr.Dropdown(value="continuous drag",
                label="task",
                choices=["continuous drag"]
            )
            fill_mode = gr.Dropdown(value="interpolation",
                label="fill mode",
                choices=["ori",
                        "0",
                        "interpolation",
                        "random"]
            )
            use_kv_cp = gr.Dropdown(value="default",
                label="consistency strategy",
                choices=["default",
                        "use",
                        "not use"]
            )
            use_lora_ = gr.Dropdown(value="not use",
                label="lora",
                choices=["default",
                        "use",
                        "not use"]
            )
        # algorithm specific parameters
        with gr.Tab("Drag Config"):
            with gr.Row():
                inversion_strength = gr.Slider(0, 1.0,
                    value=0.7,
                    label="inversion strength",
                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                start_step = gr.Number(value=0, label="start_step", precision=0, visible=False)
                start_layer = gr.Number(value=10, label="start_layer", precision=0, visible=False)
                
                local_models_dir = local_models_dir
                local_models_choice = \
                    [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                    label="Diffusion Model Path",
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "gsdf/Counterfeit-V2.5",
                        "stablediffusionapi/anything-v5",
                        "SG161222/Realistic_Vision_V2.0",
                    ] + local_models_choice
                )
                lcm_model_path = gr.Dropdown(value="SimianLuo/LCM_Dreamshaper_v7",
                    label="LCM Model Path",
                    choices=[
                        "SimianLuo/LCM_Dreamshaper_v7",
                    ] + local_models_choice
                )
                vae_path = gr.Dropdown(value="default",
                    label="VAE choice",
                    choices=["default",
                    "stabilityai/sd-vae-ft-mse"] + local_models_choice
                )

        with gr.Tab("LoRA Parameters"):
            with gr.Row():
                lora_step = gr.Number(value=80, label="LoRA training steps", precision=0)
                lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
                lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)




    # object moving
    with gr.Tab(label="Object Moving"):
        mask_om = gr.State(value=None) # store mask
        mask_fill = gr.State(value=None) # fill mask
        selected_points_om = gr.State([]) # store points
        original_image_om = gr.State(value=None) # store original input image
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                canvas_om = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                    show_label=True, height=LENGTH_OM, width=LENGTH_OM) # for mask painting
                train_lora_button_om = gr.Button("Train LoRA (optional)")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                input_image_om = gr.Image(type="numpy", label="Click Points",
                    show_label=True, height=LENGTH_OM, width=LENGTH_OM, interactive=False) # for points clicking
                undo_button_om = gr.Button("Undo point")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Region used for filling</p>""")
                # regiomUF_om
                regiomUF_om = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                    show_label=True, height=LENGTH_OM, width=LENGTH_OM, interactive=False) # for mask painting
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
                output_image_om = gr.Image(type="numpy", label="Editing Results",
                    show_label=True, height=LENGTH_OM, width=LENGTH_OM, interactive=False)
                with gr.Row():
                    run_button_om = gr.Button("Run")
                    clear_all_button_om = gr.Button("Clear All")

        # general parameters
        with gr.Row():
            prompt_om = gr.Textbox(label="prompt (optional)")
            lora_path_om = gr.Textbox(value="./lora_tmp", label="LoRA path (optional)")
            lora_status_bar_om = gr.Textbox(label="display LoRA training status")
            n_inference_step_om = gr.Textbox(value="10", label="n_inference_step") # add
            task_cat_om = gr.Dropdown(value="object moving",
                label="task",
                choices=["object moving",
                         "object copy"
                        ]
            )
            fill_mode_om = gr.Dropdown(value="interpolation",
                label="fill mode",
                choices=["ori",
                        "0",
                        "interpolation",
                        "random"]
            )
            use_kv_cp_om = gr.Dropdown(value="default",
                label="consistency strategy",
                choices=["default",
                        "use",
                        "not use"]
            )
            use_lora__om = gr.Dropdown(value="not use",
                label="lora",
                choices=["default",
                        "use",
                        "not use"]
            )
        # algorithm specific parameters
        with gr.Tab("Drag Config"):
            with gr.Row():
                inversion_strength_om = gr.Slider(0, 1.0,
                    value=0.7,
                    label="inversion strength",
                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                start_step_om = gr.Number(value=0, label="start_step_om", precision=0, visible=False)
                start_layer_om = gr.Number(value=10, label="start_layer_om", precision=0, visible=False)

                local_models_dir_om = local_models_dir
                local_models_choice = \
                    [os.path.join(local_models_dir_om,d) for d in os.listdir(local_models_dir_om) if os.path.isdir(os.path.join(local_models_dir_om,d))]
                model_path_om = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                    label="Diffusion Model Path",
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "gsdf/Counterfeit-V2.5",
                        "stablediffusionapi/anything-v5",
                        "SG161222/Realistic_Vision_V2.0",
                    ] + local_models_choice
                )
                lcm_model_path_om = gr.Dropdown(value="SimianLuo/LCM_Dreamshaper_v7",
                    label="LCM Model Path",
                    choices=[
                        "SimianLuo/LCM_Dreamshaper_v7",
                    ] + local_models_choice
                )
                vae_path_om = gr.Dropdown(value="default",
                    label="VAE choice",
                    choices=["default",
                    "stabilityai/sd-vae-ft-mse"] + local_models_choice
                )

        with gr.Tab("LoRA Parameters"):
            with gr.Row():
                lora_step_om = gr.Number(value=80, label="LoRA training steps", precision=0)
                lora_lr_om = gr.Number(value=0.0005, label="LoRA learning rate")
                lora_batch_size_om = gr.Number(value=4, label="LoRA batch size", precision=0)
                lora_rank_om = gr.Number(value=16, label="LoRA rank", precision=0)
                

    ##########################################################################
    # event definition
    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )
    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image],
    )
    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )
    train_lora_button.click(
        train_lora_interface,
        [original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank],
        [lora_status_bar]
    )
    run_button.click(
        run_drag,
        [original_image,
        input_image,
        mask,
        prompt,
        selected_points,
        inversion_strength,
        model_path,
        vae_path,
        lora_path,
        start_step,
        start_layer,
        n_inference_step,
        task_cat,
        fill_mode,
        use_kv_cp,
        use_lora_,
        lcm_model_path,
        ],
        [output_image]
    )
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas,
        input_image,
        output_image,
        selected_points,
        original_image,
        mask]
    )




    canvas_om.edit(
        store_img_om,
        [canvas_om],
        [original_image_om, selected_points_om, input_image_om, mask_om, regiomUF_om]
    )


    regiomUF_om.edit(
        store_img_om_fill,
        [regiomUF_om],
        [mask_fill]
    )

    input_image_om.select(
        get_points,
        [input_image_om, selected_points_om],
        [input_image_om],
    )
    undo_button_om.click(
        undo_points,
        [original_image_om, mask_om],
        [input_image_om, selected_points_om]
    )
    train_lora_button_om.click(
        train_lora_interface, 
        [original_image_om,
        prompt_om,
        model_path_om,
        vae_path_om,
        lora_path_om,
        lora_step_om,
        lora_lr_om,
        lora_batch_size_om,
        lora_rank_om],
        [lora_status_bar_om]
    )
    run_button_om.click(
        run_drag,
        [original_image_om,
        input_image_om,
        mask_om,
        prompt_om,
        selected_points_om,
        inversion_strength_om,
        model_path_om,
        vae_path_om,
        lora_path_om,
        start_step_om,
        start_layer_om,
        n_inference_step_om,
        task_cat_om,
        fill_mode_om,
        use_kv_cp_om,
        use_lora__om,
        lcm_model_path_om,
        mask_fill,
        ],
        [output_image_om]
    )
    clear_all_button_om.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas_om,
        input_image_om,
        output_image_om,
        selected_points_om,
        original_image_om,
        mask_om]
    )


demo.queue().launch(share=True, debug=True)
