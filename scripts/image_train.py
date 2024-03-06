"""
Train a diffusion model on images.
"""

import argparse
import os
from GAN_modules.networks import define_image_D as define_D, GANLoss
import torch


from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

   # Ensure the log directory exists
    log_dir = "/content/guided-diffusion/logs"  # Replace with your desired log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure the logger with the log directory
    logger.configure(dir=log_dir)
    #logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    # Instantiate the discriminator here
    input_nc = 3  # Assuming RGB images
    ndf = 64  # Number of discriminator filters in the first conv layer
    n_layers_D = 3  # Number of layers in the discriminator
    use_sigmoid = False  # For LSGAN, set this to False; for vanilla GAN, set to True if needed
    gpu_ids = [0]  # Adjust based on your setup

    discriminator = define_D(input_nc, ndf, 'n_layers', n_layers_D=n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    
    # After discriminator instantiation
    print("Discriminator Architecture:")
    print(discriminator)

    # Optionally, to see a more detailed summary including the number of parameters per layer
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Total parameters in discriminator: {total_params}")
    # Placeholder for discriminator training step
    # TODO: Implement discriminator training logic


    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
