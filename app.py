import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig

import aoti


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 80

MIN_DURATION = round(MIN_FRAMES_MODEL/FIXED_FPS,1)
MAX_DURATION = round(MAX_FRAMES_MODEL/FIXED_FPS,1)


pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained('cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    transformer_2=WanTransformer3DModel.from_pretrained('cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer_2',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    torch_dtype=torch.bfloat16,
).to('cuda')

pipe.load_lora_weights(
    "Kijai/WanVideo_comfy", 
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
    adapter_name="lightx2v"
)
kwargs_lora = {}
kwargs_lora["load_into_transformer_2"] = True
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy", 
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
    adapter_name="lightx2v_2", **kwargs_lora
)
pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])
pipe.unload_lora_weights()

quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

aoti.aoti_blocks_load(pipe.transformer, 'zerogpu-aoti/Wan2', variant='fp8da')
aoti.aoti_blocks_load(pipe.transformer_2, 'zerogpu-aoti/Wan2', variant='fp8da')


default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "$$\text blurry, low-res, low quality, bad anatomy, bad hands, missing limbs, extra fingers, mutated hands, deformed, disfigured, text, watermark, jpeg artifacts, tiling, duplicate, ugly}$$"

def resize_image(image: Image.Image) -> Image.Image:
    """
    Resizes an image to fit within the model's constraints, preserving aspect ratio as much as possible.
    """
    width, height = image.size

    # Handle square case
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height
    
    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM 
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM 

    image_to_resize = image
    
    if aspect_ratio > MAX_ASPECT_RATIO:
        # Very wide image -> crop width to fit 832x480 aspect ratio
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        # Very tall image -> crop height to fit 480x832 aspect ratio
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:  # Landscape
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:  # Portrait
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))
    
    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(
        int(round(duration_seconds * FIXED_FPS)),
        MIN_FRAMES_MODEL,
        MAX_FRAMES_MODEL,
    ))


def get_duration(
    input_image,
    prompt,
    steps,
    negative_prompt,
    duration_seconds,
    guidance_scale,
    guidance_scale_2,
    seed,
    randomize_seed,
    progress,
):
    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624
    BASE_STEP_DURATION = 15
    width, height = resize_image(input_image).size
    frames = get_num_frames(duration_seconds)
    factor = frames * width * height / BASE_FRAMES_HEIGHT_WIDTH
    step_duration = BASE_STEP_DURATION * factor ** 1.5
    return 10 + int(steps) * step_duration

@spaces.GPU(duration=get_duration)
def generate_video(
    input_image,
    prompt,
    steps = 4,
    negative_prompt=default_negative_prompt,
    duration_seconds = MAX_DURATION,
    guidance_scale = 1,
    guidance_scale_2 = 1,    
    seed = 42,
    randomize_seed = False,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate a video from an input image using the Wan 2.2 14B I2V model with Lightning LoRA.
    
    This function takes an input image and generates a video animation based on the provided
    prompt and parameters. It uses an FP8 qunatized Wan 2.2 14B Image-to-Video model in with Lightning LoRA
    for fast generation in 4-8 steps.
    
    Args:
        input_image (PIL.Image): The input image to animate. Will be resized to target dimensions.
        prompt (str): Text prompt describing the desired animation or motion.
        steps (int, optional): Number of inference steps. More steps = higher quality but slower.
            Defaults to 4. Range: 1-30.
        negative_prompt (str, optional): Negative prompt to avoid unwanted elements. 
            Defaults to default_negative_prompt (contains unwanted visual artifacts).
        duration_seconds (float, optional): Duration of the generated video in seconds.
            Defaults to 2. Clamped between MIN_FRAMES_MODEL/FIXED_FPS and MAX_FRAMES_MODEL/FIXED_FPS.
        guidance_scale (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        guidance_scale_2 (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        seed (int, optional): Random seed for reproducible results. Defaults to 42.
            Range: 0 to MAX_SEED (2147483647).
        randomize_seed (bool, optional): Whether to use a random seed instead of the provided seed.
            Defaults to False.
        progress (gr.Progress, optional): Gradio progress tracker. Defaults to gr.Progress(track_tqdm=True).
    
    Returns:
        tuple: A tuple containing:
            - video_path (str): Path to the generated video file (.mp4)
            - current_seed (int): The seed used for generation (useful when randomize_seed=True)
    
    Raises:
        gr.Error: If input_image is None (no image uploaded).
    
    Note:
        - Frame count is calculated as duration_seconds * FIXED_FPS (24)
        - Output dimensions are adjusted to be multiples of MOD_VALUE (32)
        - The function uses GPU acceleration via the @spaces.GPU decorator
        - Generation time varies based on steps and duration (see get_duration function)
    """
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    
    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

    return video_path, current_seed

with gr.Blocks() as demo:
    gr.Markdown("# Fast 4 steps Wan 2.2 I2V (14B) with Lightning LoRA")
    gr.Markdown("run Wan 2.2 in just 4-8 steps, with [Lightning LoRA](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Wan22-Lightning), fp8 quantization & AoT compilation - compatible with 🧨 diffusers and ZeroGPU⚡️")
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")
            
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps") 
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")

            generate_button = gr.Button("Generate Video", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, interactive=False)
    
    ui_inputs = [
        input_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox
    ]
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input])

    gr.Examples(
        examples=[ 
            [
                "wan_i2v_input.JPG",
                "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind (clear water, green hills, blue sky with clouds). Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams. Brief underwater view of cat’s face, then cat resurfaces, still filming selfie, playful summer vacation mood.",
                4,
            ],
            [
                "wan22_input_2.jpg",
                "A sleek lunar vehicle glides into view from left to right, kicking up moon dust as astronauts in white spacesuits hop aboard with characteristic lunar bouncing movements. In the distant background, a VTOL craft descends straight down and lands silently on the surface. Throughout the entire scene, ethereal aurora borealis ribbons dance across the star-filled sky, casting shimmering curtains of green, blue, and purple light that bathe the lunar landscape in an otherworldly, magical glow.",
                4,
            ],
            [
                "kill_bill.jpeg",
                "Uma Thurman's character, Beatrix Kiddo, holds her razor-sharp katana blade steady in the cinematic lighting. Suddenly, the polished steel begins to soften and distort, like heated metal starting to lose its structural integrity. The blade's perfect edge slowly warps and droops, molten steel beginning to flow downward in silvery rivulets while maintaining its metallic sheen. The transformation starts subtly at first - a slight bend in the blade - then accelerates as the metal becomes increasingly fluid. The camera holds steady on her face as her piercing eyes gradually narrow, not with lethal focus, but with confusion and growing alarm as she watches her weapon dissolve before her eyes. Her breathing quickens slightly as she witnesses this impossible transformation. The melting intensifies, the katana's perfect form becoming increasingly abstract, dripping like liquid mercury from her grip. Molten droplets fall to the ground with soft metallic impacts. Her expression shifts from calm readiness to bewilderment and concern as her legendary instrument of vengeance literally liquefies in her hands, leaving her defenseless and disoriented.",
                6,
            ],
        ],
        inputs=[input_image_component, prompt_input, steps_slider], outputs=[video_output, seed_input], fn=generate_video, cache_examples="lazy"
    )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)