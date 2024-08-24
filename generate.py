import os
import torch
import argparse
from diffusers import DiffusionPipeline

from handler import WebhookHandler

def load_prompts_from_file(file_path):
    prompts_by_size = {}
    current_size = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                current_size = line[2:].strip()
                prompts_by_size[current_size] = []
            elif line:
                prompts_by_size[current_size].append(line)
    return prompts_by_size

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate images using a diffusion pipeline")
    parser.add_argument("--model_id", type=str, default=os.getenv('MODEL_ID', 'black-forest-labs/FLUX.1-dev'), help="Model ID to use for the pipeline")
    parser.add_argument("--adapter_id", type=str, default=os.getenv('ADAPTER_ID', 'tungdop2/FLUX.1-dev-dreambooth-veronika_512'), help="Adapter ID to load LoRA weights")
    parser.add_argument("--output_dir", type=str, default=os.getenv('OUTPUT_DIR', 'tmp/old'), help="Directory to save the generated images")
    parser.add_argument("--steps", type=int, default=int(os.getenv('STEPS', 50)), help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=float(os.getenv('GUIDANCE_SCALE', 3.5)), help="Guidance scale for image generation")
    parser.add_argument("--max_sequence_length", type=int, default=int(os.getenv('MAX_SEQUENCE_LENGTH', 512)), help="Maximum sequence length for the prompt")
    parser.add_argument("--config_path", type=str, default=os.getenv('CONFIG_PATH', 'webhook.json'), help="Path to the webhook configuration file")
    parser.add_argument("--prompts_file", type=str, default=os.getenv('PROMPTS_FILE', 'prompts.txt'), help="Path to the prompts file")

    args = parser.parse_args()
    
    handler = WebhookHandler(
        config_path=args.config_path
    )
    handler.send("Starting inference")

    # Load prompts from file
    handler.send("Loading prompts from file")
    prompts_by_size = load_prompts_from_file(args.prompts_file)

    # Initialize pipeline
    handler.send("Initializing pipeline")
    pipeline = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipeline.load_lora_weights(args.adapter_id)

    handler.send("Moving pipeline to available device")
    pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    pipeline.enable_model_cpu_offload()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Send generation configs
    config_message = "Generation configurations: \
        \nModel ID: `{args.model_id}` \
        \nAdapter ID: `{args.adapter_id}` \
        \nOutput directory: `{args.output_dir}` \
        \nNumber of inference steps: `{args.steps}` \
        \nGuidance scale: `{args.guidance_scale}` \
        \nMaximum sequence length: `{args.max_sequence_length}`"
    handler.send(config_message)

    for size, prompts in prompts_by_size.items():
        sizedir = f"{args.output_dir}/{size}"
        os.makedirs(sizedir, exist_ok=True)
        h, w = map(int, size.split('x'))
        for prompt_idx, prompt in enumerate(prompts):
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.steps,
                width=w,
                height=h,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
            ).images[0]
            image.save(f"{sizedir}/{prompt_idx}_{prompt[:40].replace(' ', '_')}.jpg", format='jpeg')
            handler.send(f"Generated image {prompt_idx} for size {size}", images=image)

if __name__ == "__main__":
    main()
