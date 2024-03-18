import argparse
import os
import json
import numpy as np
import torch as th
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, add_dict_to_argparser, args_to_dict

def load_bbs(bb_dir):
    bbs = []
    for bb_name in sorted(os.listdir(bb_dir)):
        if bb_name.endswith('.json'):
            bb_path = os.path.join(bb_dir, bb_name)
            with open(bb_path, 'r') as f:
                bb = json.load(f)
            bbs.append(bb)
    return bbs

def apply_noise_within_bbs(shape, bb):
    noisy_image = th.randn(shape)
    intense_noise = th.randn_like(noisy_image[:, bb['y']:bb['y']+bb['h'], bb['x']:bb['x']+bb['w']]) * 2
    noisy_image[:, bb['y']:bb['y']+bb['h'], bb['x']:bb['x']+bb['w']] = intense_noise
    return noisy_image

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading BBs...")
    bbs = load_bbs("/content/guided-diffusion/sampling/bbox")

    logger.log("sampling...")
    all_images = []
    save_interval = 32  # Save after every 16 samples

    for i in range(args.num_samples):
        bb = bbs[i % len(bbs)]
        noise_shape = (3, 256, 256)
        noisy_image = apply_noise_within_bbs(noise_shape, bb).to(dist_util.dev())
        noisy_image = noisy_image.unsqueeze(0)  # Add a batch dimension
        sample = diffusion.p_sample_loop(model, noisy_image.shape, noise=noisy_image, clip_denoised=args.clip_denoised)
        sample_denorm = ((sample + 1) / 2 * 255).clamp(0, 255).to(th.uint8)
        image_np = sample_denorm[0].cpu().numpy().transpose(1, 2, 0)
        all_images.append(image_np)
        logger.log(f"created {len(all_images)} samples")

        # Check if it's time to save
        if (i + 1) % save_interval == 0 or i == args.num_samples - 1:
            # Convert list of images to a 4D numpy array (N x H x W x C)
            arr = np.stack(all_images).astype(np.uint8)
            batch_num = (i + 1) // save_interval
            #out_path = f"/content/guided-diffusion/results/samples_batch_{batch_num}.npz"
            out_path = f"/content/guided-diffusion/sample_images/samples_batch_{batch_num}.npz"
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)
            all_images = []  # Reset the list for the next batch

    logger.log("sampling complete")

def create_argparser():
    defaults = dict(clip_denoised=True, num_samples=16, batch_size=16, use_ddim=False, model_path="")
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
