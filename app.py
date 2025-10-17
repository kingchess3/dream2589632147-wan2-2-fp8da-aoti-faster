import spaces

import torch

import time

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



# الثوابت الرئيسية للأبعاد

MAX_DIM = 832

MIN_DIM = 480

SQUARE_DIM = 640

MULTIPLE_OF = 16



# الأبعاد المستهدفة لتناسب السوشيال ميديا (العرض/الارتفاع)

ASPECT_RATIOS = {

    "Auto": None,

    "16:9 (Landscape)": 16/9,

    "1:1 (Square)": 1/1,

    "4:5 (Portrait)": 4/5,

    "9:16 (Vertical)": 9/16,

}

DEFAULT_RATIO_KEY = list(ASPECT_RATIOS.keys())[0]





MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16

MIN_FRAMES_MODEL = 8

MAX_FRAMES_MODEL = 112 # تم التعديل ليكون الحد الأقصى 7 ثوانٍ (112 / 16 = 7.0)



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

default_negative_prompt = "blurry, low-res, low quality, bad anatomy, bad hands, missing limbs, extra fingers, mutated hands, deformed, disfigured, text, watermark, jpeg artifacts, tiling, duplicate, ugly"



def resize_image(image: Image.Image, target_ratio_key: str = DEFAULT_RATIO_KEY) -> Image.Image:

    """

    Resizes and crops an image to fit the model's constraints and the target aspect ratio,

    preserving aspect ratio as much as possible before final scaling.

    """

    width, height = image.size

    target_ratio = ASPECT_RATIOS.get(target_ratio_key)



    image_to_resize = image

    

    # 1. تطبيق القص (Cropping) بناءً على النسبة المطلوبة

    if target_ratio is not None:

        current_ratio = width / height

        

        if current_ratio > target_ratio:

            # الصورة أوسع مما يجب: قص العرض

            crop_width = int(round(height * target_ratio))

            left = (width - crop_width) // 2

            image_to_resize = image.crop((left, 0, left + crop_width, height))

            width, height = image_to_resize.size # تحديث الأبعاد

        elif current_ratio < target_ratio:

            # الصورة أطول مما يجب: قص الارتفاع

            crop_height = int(round(width / target_ratio))

            top = (height - crop_height) // 2

            image_to_resize = image.crop((0, top, width, top + crop_height))

            width, height = image_to_resize.size # تحديث الأبعاد



    # 2. تحديد الأبعاد النهائية للنموذج (مع مراعاة قيود MIN/MAX DIM)

    aspect_ratio = width / height

    

    if width == height: # مربع

        target_w, target_h = SQUARE_DIM, SQUARE_DIM

    elif aspect_ratio > 1: # أفقي (Landscape)

        target_w = MAX_DIM

        target_h = int(round(target_w / aspect_ratio))

    else: # عمودي (Portrait)

        target_h = MAX_DIM

        target_w = int(round(target_h * aspect_ratio))

    

    # ضمان أن تكون الأبعاد مضاعفاً لـ MULTIPLE_OF

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF

    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF



    # ضمان أن تكون الأبعاد ضمن الحد الأدنى والأقصى

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

    aspect_ratio_key, # تم إضافة النسبة

    progress,

):

    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624

    BASE_STEP_DURATION = 15

    if input_image is None:

        return 0

    # تمرير النسبة إلى دالة resize_image

    width, height = resize_image(input_image, aspect_ratio_key).size

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

    aspect_ratio_key = DEFAULT_RATIO_KEY, # تم إضافة النسبة

    progress=gr.Progress(track_tqdm=True),

):

    """

    Generate a video from an input image using the Wan 2.2 14B I2V model with Lightning LoRA.

    """

    if input_image is None:

        raise gr.Error("Please upload an input image.")

    

    start_time = time.time()



    num_frames = get_num_frames(duration_seconds)

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    

    # تمرير النسبة إلى دالة resize_image

    resized_image = resize_image(input_image, aspect_ratio_key)



    try:

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

    except torch.cuda.OutOfMemoryError:

        raise gr.Error("Error: GPU memory exhausted (Out of Memory). Try reducing duration or image size.")

    except Exception as e:

        raise gr.Error(f"An unexpected error occurred during generation: {e}")



    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:

        video_path = tmpfile.name



    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

    

    # إدارة الذاكرة

    del output_frames_list

    torch.cuda.empty_cache()

    gc.collect()



    end_time = time.time()

    duration_str = f"Video generated in {end_time - start_time:.2f} seconds. Seed used: {current_seed}"

    print(duration_str)

    

    return video_path, current_seed



def get_image_info(image, aspect_ratio_key):

    if image is None:

        return ""

    # تمرير النسبة إلى دالة resize_image

    resized_image = resize_image(image, aspect_ratio_key)

    return f"{resized_image.width}x{resized_image.height}"





with gr.Blocks() as demo:

    gr.Markdown("# Fast 4 steps Wan 2.2 I2V (14B) with Lightning LoRA")

    gr.Markdown("run Wan 2.2 in just 4-8 steps, with [Lightning LoRA](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Wan22-Lightning), fp8 quantization & AoT compilation - compatible with 🧨 diffusers and ZeroGPU⚡️")

    with gr.Row():

        with gr.Column():

            input_image_component = gr.Image(type="pil", label="Input Image")

            

            # إضافة قائمة منسدلة لنسبة العرض إلى الارتفاع

            aspect_ratio_input = gr.Dropdown(

                label="Target Aspect Ratio (Social Media Formats)",

                choices=list(ASPECT_RATIOS.keys()),

                value=DEFAULT_RATIO_KEY,

                info="Crops the image to fit the selected social media ratio before generation."

            )

            

            image_dim_output = gr.Textbox(label="Target Dimensions (W x H)", value="", interactive=False)

            

            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=7.0, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")

            

            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)

            

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

            

    

    # ربط حدث تحميل الصورة أو تغيير نسبة الأبعاد بعرض الأبعاد المستهدفة

    input_image_component.change(fn=get_image_info, inputs=[input_image_component, aspect_ratio_input], outputs=[image_dim_output], queue=False)

    aspect_ratio_input.change(fn=get_image_info, inputs=[input_image_component, aspect_ratio_input], outputs=[image_dim_output], queue=False)

    

    ui_inputs = [

        input_image_component, prompt_input, steps_slider,

        negative_prompt_input, duration_seconds_input,

        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox,

        aspect_ratio_input # تم إضافة نسبة الأبعاد إلى المدخلات

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