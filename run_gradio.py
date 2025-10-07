from typing import List
import math
from argparse import ArgumentParser
import random
import os
import signal
import sys
import glob
import threading
import time
from datetime import datetime

import numpy as np
import torch
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate.utils import set_seed

from diffbir.model.cldm import ControlLDM
from diffbir.model.swinir import SwinIR
from diffbir.inference.pretrained_models import MODELS
from diffbir.utils.common import instantiate_from_config, load_model_from_url
from diffbir.model.gaussian_diffusion import Diffusion
from diffbir.pipeline import SwinIRPipeline
from diffbir.utils.caption import (
    EmptyCaptioner,
    LLaVACaptioner,
    RAMCaptioner,
    LLAVA_AVAILABLE,
    RAM_AVAILABLE,
)

torch.set_grad_enabled(False)

# Signal handler for Ctrl+C graceful shutdown
def signal_handler(sig, frame):
    print('\n\n🛑 Received Ctrl+C! Shutting down gracefully...')
    print('Cleaning up resources...')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# This gradio script only support DiffBIR v2.1
parser = ArgumentParser()
parser.add_argument("--captioner", type=str, choices=["none", "ram", "llava"], required=True)
parser.add_argument("--llava_bit", type=str, choices=["4", "8", "16"], default="4")
parser.add_argument("--share", action="store_true", help="Enable sharing via gradio link")
parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name/IP to bind to")
parser.add_argument("--server-port", type=int, default=8860, help="Server port to bind to")
args = parser.parse_args()

# Set max height and width to constraint inference time for online demo
max_height = 15048
max_width = 15048

# Create output directories if they don't exist
output_dir = "output"
batch_output_dir = os.path.join(output_dir, "batches")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(batch_output_dir, exist_ok=True)

tasks = ["sr", "face"]
device = "cuda"
precision = "fp16"
llava_bit = args.llava_bit
# Set captioner to llava or ram to enable auto-caption
captioner = args.captioner

if captioner == "llava":
    assert LLAVA_AVAILABLE
elif captioner == "ram":
    assert RAM_AVAILABLE

# 1. load stage-1 models
swinir: SwinIR = instantiate_from_config(
    OmegaConf.load("configs/inference/swinir.yaml")
)
swinir.load_state_dict(load_model_from_url(MODELS["swinir_realesrgan"]))
swinir.eval().to(device)

face_swinir: SwinIR = instantiate_from_config(
    OmegaConf.load("configs/inference/swinir.yaml")
)
face_swinir.load_state_dict(load_model_from_url(MODELS["swinir_face"]))
face_swinir.eval().to(device)

# 2. load stage-2 model
cldm: ControlLDM = instantiate_from_config(
    OmegaConf.load("configs/inference/cldm.yaml")
)
# 2.1 load pre-trained SD
sd_weight = load_model_from_url(MODELS["sd_v2.1_zsnr"])
unused, missing = cldm.load_pretrained_sd(sd_weight)
print(
    f"load pretrained stable diffusion, "
    f"unused weights: {unused}, missing weights: {missing}"
)
# 2.2 load ControlNet
control_weight = load_model_from_url(MODELS["v2.1"])
cldm.load_controlnet_from_ckpt(control_weight)
print("load controlnet weight")
cldm.eval().to(device)
cast_type = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}[precision]
cldm.cast_dtype(cast_type)

# 3. load noise schedule
diffusion: Diffusion = instantiate_from_config(
    OmegaConf.load("configs/inference/diffusion_v2.1.yaml")
)
diffusion.to(device)

# 4. load captioner
if captioner == "none":
    captioner = EmptyCaptioner(device)
elif captioner == "llava":
    captioner = LLaVACaptioner(device, llava_bit)
else:
    captioner = RAMCaptioner(device)

error_image = np.array(Image.open("assets/gradio_error_img.png"))

def extract_original_filename(file_path):
    """Extract original filename from Gradio's file path with better handling"""
    try:
        if hasattr(file_path, 'name'):
            # If it's a file object, get the path
            temp_path = file_path.name
        elif hasattr(file_path, 'orig_name'):
            # Gradio sometimes stores original name
            return os.path.splitext(file_path.orig_name)[0]
        else:
            # If it's already a string path
            temp_path = file_path
            
        # Try to get the original name from the temp path
        filename = os.path.basename(temp_path)
        base_name = os.path.splitext(filename)[0]
        
        # If it's a typical temp name (like tmpXXXXXX), generate a timestamp-based name
        if base_name.startswith('tmp') or len(base_name) < 3:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"image_{timestamp}"
            
        print(f"Extracted filename: {base_name} from path: {temp_path}")
        return base_name
    except Exception as e:
        print(f"Error extracting filename: {e}")
        # Generate a unique timestamp-based name as fallback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        return f"image_{timestamp}"

def generate_unique_filename(base_name, upscale, steps, output_dir, is_batch=False):
    """Generate a unique filename to avoid overwriting existing files"""
    base_filename = f"{base_name}-{upscale}x-{steps}s"
    extension = ".png"
    
    if is_batch:
        filepath = os.path.join(output_dir, f"{base_filename}{extension}")
    else:
        filepath = os.path.join(output_dir, f"{base_filename}{extension}")
    
    # If file doesn't exist, use the base name
    if not os.path.exists(filepath):
        return filepath, f"{base_filename}{extension}"
    
    # If file exists, add a counter
    counter = 1
    while True:
        new_filename = f"{base_filename}_{counter:03d}{extension}"
        new_filepath = os.path.join(output_dir, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath, new_filename
        counter += 1
        # Safety check to prevent infinite loop
        if counter > 999:
            # Use timestamp as last resort
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
            final_filename = f"{base_filename}_{timestamp}{extension}"
            final_filepath = os.path.join(output_dir, final_filename)
            return final_filepath, final_filename

def save_image_with_custom_name(image_array, upscale, steps, is_batch=False, original_filename=None, batch_dir=None):
    """Save the processed image with custom naming format and unique filename"""
    try:
        # Convert numpy array to PIL Image
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        # Use provided filename or generate a timestamp-based one
        if original_filename:
            base_name = original_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            base_name = f"image_{timestamp}"
        
        # Determine output directory
        if is_batch and batch_dir:
            target_dir = batch_dir
        elif is_batch:
            target_dir = batch_output_dir
        else:
            target_dir = output_dir
        
        # Generate unique filename
        filepath, filename = generate_unique_filename(base_name, upscale, steps, target_dir, is_batch)
        
        # Save the image
        image.save(filepath, "PNG")
        print(f"Image saved to: {filepath}")
        return filepath, filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None, None

def process_batch_images(
    input_files,
    task,
    upscale,
    cleaner_tiled,
    cleaner_tile_size,
    vae_encoder_tiled,
    vae_encoder_tile_size,
    vae_decoder_tiled,
    vae_decoder_tile_size,
    cldm_tiled,
    cldm_tile_size,
    positive_prompt,
    negative_prompt,
    cfg_scale,
    rescale_cfg,
    strength,
    noise_aug,
    steps,
    sampler_type,
    s_churn,
    s_tmin,
    s_tmax,
    s_noise,
    order,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    """Process multiple images with the same settings"""
    if not input_files:
        return [], "No images uploaded for batch processing"
    
    results = []
    status_messages = []
    failed_count = 0
    success_count = 0
    
    # Create a unique batch folder with timestamp
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_batch_dir = os.path.join(batch_output_dir, f"batch_{batch_timestamp}")
    os.makedirs(current_batch_dir, exist_ok=True)
    
    total_files = len(input_files)
    
    for i, file_info in enumerate(progress.tqdm(input_files, desc="Processing batch")):
        try:
            # Load image and extract original filename
            input_image = Image.open(file_info.name).convert('RGB')
            original_filename = extract_original_filename(file_info)
            
            # Use the same seed for all images (or generate new ones if seed is -1)
            current_seed = seed
            if seed == -1:
                current_seed = random.randint(0, 2147483647)
            
            set_seed(current_seed)
            
            # Process the image using the same logic as single processing
            lq = input_image
            caption = captioner(lq)
            pos_prompt = ", ".join([text for text in [caption, positive_prompt] if text])
            neg_prompt = negative_prompt
            
            # Upscale and convert to numpy array
            out_w, out_h = tuple(int(x * upscale) for x in lq.size)
            if out_w > max_width or out_h > max_height:
                status_messages.append(f"❌ {original_filename}: Resolution too large ({out_h}x{out_w})")
                failed_count += 1
                continue
                
            lq = lq.resize((out_w, out_h), Image.BICUBIC)
            lq = np.array(lq)
            
            # Select cleaner
            if task == "sr":
                cleaner = swinir
            else:
                cleaner = face_swinir
                
            # Create pipeline
            pipeline = SwinIRPipeline(cleaner, cldm, diffusion, None, device)
            
            # Run pipeline
            sample = pipeline.run(
                lq[None],
                steps,
                strength,
                cleaner_tiled,
                cleaner_tile_size,
                cleaner_tile_size // 2,
                vae_encoder_tiled,
                vae_encoder_tile_size,
                vae_decoder_tiled,
                vae_decoder_tile_size,
                cldm_tiled,
                cldm_tile_size,
                cldm_tile_size // 2,
                pos_prompt,
                neg_prompt,
                cfg_scale,
                "noise",
                sampler_type,
                noise_aug,
                rescale_cfg,
                s_churn,
                s_tmin,
                s_tmax,
                s_noise,
                1,
                order,
            )[0]
            
            # Save to batch folder with custom naming
            filepath, filename = save_image_with_custom_name(
                sample, upscale, steps, is_batch=True, 
                original_filename=original_filename, batch_dir=current_batch_dir
            )
            
            if filepath:
                results.append(sample)
                status_messages.append(f"✅ {original_filename}: Saved as {filename}")
                success_count += 1
            else:
                status_messages.append(f"⚠️ {original_filename}: Processed but failed to save")
                results.append(sample)
                success_count += 1
            
        except Exception as e:
            status_messages.append(f"❌ {original_filename if 'original_filename' in locals() else 'Unknown'}: Failed - {str(e)}")
            failed_count += 1
            continue
    
    # Create summary
    summary = f"Batch processing completed!\n"
    summary += f"📁 Batch folder: {current_batch_dir}\n"
    summary += f"✅ Successful: {success_count}/{total_files}\n"
    summary += f"❌ Failed: {failed_count}/{total_files}\n\n"
    summary += "Details:\n" + "\n".join(status_messages)
    
    return results, summary

@torch.no_grad()
def process(
    input_image,
    task,
    upscale,
    cleaner_tiled,
    cleaner_tile_size,
    vae_encoder_tiled,
    vae_encoder_tile_size,
    vae_decoder_tiled,
    vae_decoder_tile_size,
    cldm_tiled,
    cldm_tile_size,
    positive_prompt,
    negative_prompt,
    cfg_scale,
    rescale_cfg,
    strength,
    noise_aug,
    steps,
    sampler_type,
    s_churn,
    s_tmin,
    s_tmax,
    s_noise,
    order,
    seed,
    progress=gr.Progress(track_tqdm=True),
) -> List[np.ndarray]:
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    set_seed(seed)
    lq = input_image
    
    # Generate a unique filename for single image processing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    current_filename = f"image_{timestamp}"
    
    # Prepare prompt
    caption = captioner(lq)
    pos_prompt = ", ".join([text for text in [caption, positive_prompt] if text])
    neg_prompt = negative_prompt
    # Upscale and convert to numpy array
    out_w, out_h = tuple(int(x * upscale) for x in lq.size)
    if out_w > max_width or out_h > max_height:
        return [error_image], (
            "Failed :( The requested resolution exceeds the maximum limit. "
            f"Your requested resolution is ({out_h}, {out_w}). "
            f"The maximum allowed resolution is ({max_height}, {max_width})."
        )
    lq = lq.resize((out_w, out_h), Image.BICUBIC)
    lq = np.array(lq)
    # Select cleaner
    if task == "sr":
        cleaner = swinir
    else:
        cleaner = face_swinir
    # Create pipeline
    pipeline = SwinIRPipeline(cleaner, cldm, diffusion, None, device)
    # Run pipeline to restore this image
    try:
        sample = pipeline.run(
            lq[None],
            steps,
            strength,
            cleaner_tiled,
            cleaner_tile_size,
            cleaner_tile_size // 2,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            cldm_tile_size // 2,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            "noise",
            sampler_type,
            noise_aug,
            rescale_cfg,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            1,
            order,
        )[0]
        
        # Save the output image with custom naming
        saved_path, filename = save_image_with_custom_name(
            sample, upscale, steps, is_batch=False, original_filename=current_filename
        )
        
        if saved_path:
            success_message = f"Success! Image saved as: {filename}"
        else:
            success_message = "Success! (Warning: Could not save image)"
        
        return [sample], success_message
    except Exception as e:
        return [error_image], f"Failed :( {e}"

# Store uploaded file info globally to extract filename
uploaded_file_info = {}

def handle_file_upload(file):
    """Handle file upload and store file info for filename extraction"""
    global uploaded_file_info
    if file is not None:
        # Store the file info with a timestamp key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uploaded_file_info[timestamp] = {
            'file': file,
            'original_name': extract_original_filename(file)
        }
        print(f"File uploaded: {uploaded_file_info[timestamp]['original_name']}")
    return file

def process_with_filename(
    input_image,
    task,
    upscale,
    cleaner_tiled,
    cleaner_tile_size,
    vae_encoder_tiled,
    vae_encoder_tile_size,
    vae_decoder_tiled,
    vae_decoder_tile_size,
    cldm_tiled,
    cldm_tile_size,
    positive_prompt,
    negative_prompt,
    cfg_scale,
    rescale_cfg,
    strength,
    noise_aug,
    steps,
    sampler_type,
    s_churn,
    s_tmin,
    s_tmax,
    s_noise,
    order,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    """Wrapper to process with better filename handling"""
    global uploaded_file_info
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    set_seed(seed)
    lq = input_image
    
    # Try to get the most recent uploaded filename
    current_filename = None
    if uploaded_file_info:
        # Get the most recent upload
        latest_key = max(uploaded_file_info.keys())
        current_filename = uploaded_file_info[latest_key]['original_name']
        # Clean up old entries (keep only the latest 5)
        if len(uploaded_file_info) > 5:
            old_keys = sorted(uploaded_file_info.keys())[:-5]
            for key in old_keys:
                del uploaded_file_info[key]
    
    if not current_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        current_filename = f"image_{timestamp}"
    
    # Prepare prompt
    caption = captioner(lq)
    pos_prompt = ", ".join([text for text in [caption, positive_prompt] if text])
    neg_prompt = negative_prompt
    # Upscale and convert to numpy array
    out_w, out_h = tuple(int(x * upscale) for x in lq.size)
    if out_w > max_width or out_h > max_height:
        return [error_image], (
            "Failed :( The requested resolution exceeds the maximum limit. "
            f"Your requested resolution is ({out_h}, {out_w}). "
            f"The maximum allowed resolution is ({max_height}, {max_width})."
        )
    lq = lq.resize((out_w, out_h), Image.BICUBIC)
    lq = np.array(lq)
    # Select cleaner
    if task == "sr":
        cleaner = swinir
    else:
        cleaner = face_swinir
    # Create pipeline
    pipeline = SwinIRPipeline(cleaner, cldm, diffusion, None, device)
    # Run pipeline to restore this image
    try:
        sample = pipeline.run(
            lq[None],
            steps,
            strength,
            cleaner_tiled,
            cleaner_tile_size,
            cleaner_tile_size // 2,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            cldm_tile_size // 2,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            "noise",
            sampler_type,
            noise_aug,
            rescale_cfg,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            1,
            order,
        )[0]
        
        # Save the output image with custom naming
        saved_path, filename = save_image_with_custom_name(
            sample, upscale, steps, is_batch=False, original_filename=current_filename
        )
        
        if saved_path:
            success_message = f"Success! Image saved as: {filename}"
        else:
            success_message = "Success! (Warning: Could not save image)"
        
        return [sample], success_message
    except Exception as e:
        return [error_image], f"Failed :( {e}"

DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera,hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
)


DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth."
)

block = gr.Blocks().queue()
with block:
    with gr.Tabs():
        with gr.TabItem("Single Image Processing"):
            # Move the Run button to the top
            run_button = gr.Button(value="🚀 Run Processing", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(sources="upload", type="pil")
                    with gr.Accordion("Basic Options", open=True):
                        with gr.Row():
                            task = gr.Dropdown(
                                label="Task",
                                choices=tasks,
                                value="sr",
                            )
                            upscale = gr.Slider(
                                label="Upsample factor",
                                minimum=1,
                                maximum=8,
                                value=1,
                                step=1,
                            )
                        with gr.Row():
                            with gr.Column():
                                cleaner_tiled = gr.Checkbox(
                                    label="Tiled cleaner",
                                    value=False,
                                )
                                cleaner_tile_size = gr.Slider(
                                    label="Cleaner tile size",
                                    minimum=256,
                                    maximum=1024,
                                    value=256,
                                    step=64,
                                )
                            with gr.Column():
                                vae_encoder_tiled = gr.Checkbox(
                                    label="Tiled VAE encoder",
                                    value=False,
                                )
                                vae_encoder_tile_size = gr.Slider(
                                    label="VAE encoder tile size",
                                    minimum=256,
                                    maximum=1024,
                                    value=256,
                                    step=64,
                                )
                        with gr.Row():
                            with gr.Column():
                                vae_decoder_tiled = gr.Checkbox(
                                    label="Tiled VAE decoder",
                                    value=False,
                                )
                                vae_decoder_tile_size = gr.Slider(
                                    label="VAE decoder tile size",
                                    minimum=256,
                                    maximum=1024,
                                    value=256,
                                    step=64,
                                )
                            with gr.Column():
                                cldm_tiled = gr.Checkbox(
                                    label="Tiled diffusion",
                                    value=True,
                                )
                                cldm_tile_size = gr.Slider(
                                    label="Diffusion tile size",
                                    minimum=512,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                )
                        seed = gr.Slider(
                            label="Seed", minimum=-1, maximum=2147483647, step=1, value=231
                        )
                    with gr.Accordion("Condition Options", open=True):
                        pos_prompt = gr.Textbox(
                            label="Positive prompt",
                            value=DEFAULT_POS_PROMPT,
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative prompt",
                            value=DEFAULT_NEG_PROMPT,
                        )
                        cfg_scale = gr.Slider(
                            label="Classifier-free guidance (cfg) scale",
                            minimum=1,
                            maximum=10,
                            value=8,
                            step=1,
                        )
                        rescale_cfg = gr.Checkbox(value=False, label="Gradually increase cfg scale")
                        with gr.Row():
                            strength = gr.Slider(
                                label="Control strength",
                                minimum=0.0,
                                maximum=1.5,
                                value=1.0,
                                step=0.1,
                            )
                            noise_aug = gr.Slider(
                                label="Noise level of condition",
                                minimum=0,
                                maximum=199,
                                value=0,
                                step=10,
                            )
                    with gr.Accordion("Sampler Options", open=True):
                        steps = gr.Slider(
                            label="Steps", minimum=5, maximum=50, value=10, step=5
                        )
                        sampler_type = gr.Dropdown(
                            label="Select a sampler",
                            choices=[
                                "dpm++_m2",
                                "spaced",
                                "ddim",
                                "edm_euler",
                                "edm_euler_a",
                                "edm_heun",
                                "edm_dpm_2",
                                "edm_dpm_2_a",
                                "edm_lms",
                                "edm_dpm++_2s_a",
                                "edm_dpm++_sde",
                                "edm_dpm++_2m",
                                "edm_dpm++_2m_sde",
                                "edm_dpm++_3m_sde",
                            ],
                            value="edm_dpm++_3m_sde",
                        )
                        s_churn = gr.Slider(
                            label="s_churn",
                            minimum=0,
                            maximum=40,
                            value=0,
                            step=1,
                        )
                        s_tmin = gr.Slider(
                            label="s_tmin",
                            minimum=0,
                            maximum=300,
                            value=0,
                            step=10,
                        )
                        s_tmax = gr.Slider(
                            label="s_tmax",
                            minimum=0,
                            maximum=300,
                            value=300,
                            step=10,
                        )
                        s_noise = gr.Slider(
                            label="s_noise",
                            minimum=1,
                            maximum=1.1,
                            value=1,
                            step=0.001,
                        )
                        order = gr.Slider(
                            label="order",
                            minimum=1,
                            maximum=8,
                            value=1,
                            step=1,
                        )
                
                with gr.Column(scale=1):
                    result_gallery = gr.Gallery(
                        label="Processing Output", 
                        show_label=True, 
                        columns=1, 
                        format="png",
                        height=400
                    )
                    status = gr.Textbox(label="Status")
        
        with gr.TabItem("Batch Processing"):
            # Move the Batch Run button to the top
            batch_run_button = gr.Button(value="🚀 Run Batch Processing", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="Upload Images for Batch Processing",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    
                    gr.Markdown("### Batch Processing Info")
                    gr.Markdown("- Upload multiple images to process them all with the same settings")
                    gr.Markdown("- All settings from the Single Image tab will be applied to each image")
                    gr.Markdown("- Results will be saved to `output/batches/batch_[timestamp]/`")
                    gr.Markdown("- Output format: `originalname-2x-10s.png` (with counter if duplicate)")
                    
                with gr.Column(scale=1):
                    batch_result_gallery = gr.Gallery(
                        label="Batch Results", 
                        show_label=False, 
                        columns=3, 
                        format="png",
                        height=400
                    )
                    batch_status = gr.Textbox(label="Batch Status", lines=10)
    
    # Event handlers
    
    # Handle file upload to extract filename
    input_image.upload(
        fn=handle_file_upload,
        inputs=[input_image],
        outputs=[input_image]
    )
    
    # Single image processing
    run_button.click(
        fn=process_with_filename,
        inputs=[
            input_image,
            task,
            upscale,
            cleaner_tiled,
            cleaner_tile_size,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            rescale_cfg,
            strength,
            noise_aug,
            steps,
            sampler_type,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            order,
            seed,
        ],
        outputs=[result_gallery, status],
    )
    
    # Batch processing
    batch_run_button.click(
        fn=process_batch_images,
        inputs=[
            batch_files,
            task,
            upscale,
            cleaner_tiled,
            cleaner_tile_size,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            rescale_cfg,
            strength,
            noise_aug,
            steps,
            sampler_type,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            order,
            seed,
        ],
        outputs=[batch_result_gallery, batch_status],
    )

print("🚀 Starting DiffBIR Gradio Interface...")
print("💡 Press Ctrl+C to gracefully shutdown the application")

# Launch with configurable options
try:
    block.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True
    )
except KeyboardInterrupt:
    print("\n🛑 Application interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error launching application: {e}")
    sys.exit(1)