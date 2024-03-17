import copy
import functools
import os
import json
import blobfile as bf
import torch as th
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.utils as vutils
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from GAN_modules.networks import define_person_D, GANLoss
import torch.optim as optim
import torchvision


from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def apply_bb_mask(tensor, mask):
    """Apply bounding box masks to a tensor."""
    return tensor * mask

def masked_loss(output, target, mask):
    """Calculate the loss only within the masked (BB) regions."""
    masked_output = apply_bb_mask(output, mask)
    masked_target = apply_bb_mask(target, mask)
    return F.mse_loss(masked_output, masked_target, reduction='mean')

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval, 
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        discriminator,  # Add the discriminator parameter
        gan_loss,       # Add the GAN loss parameter
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.discriminator = discriminator  # Store the discriminator as an attribute
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.gan_loss = gan_loss            # Store the GAN loss as an attribute
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]


        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def load_bb_data(self, image_filename, bbox_dir="/content/guided-diffusion/datasets/Cityscapes_3009_bb/"):
        bb_file = os.path.join(bbox_dir, os.path.splitext(image_filename)[0] + ".json")
        with open(bb_file, "r") as f:
            bb_data = json.load(f)
        return bb_data  # Assuming bb_data is a dictionary with 'x', 'y', 'w', 'h' keys        

    def run_loop(self):
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
           for batch_data in self.data:  # self.data is your DataLoader instance
               images, additional_info = batch_data
               filenames = additional_info['image_filenames']  # Extract filenames
               batch = images.to(dist_util.dev())  # Move batch to the correct device
               cond = {'image_filenames': filenames}  # Use the filenames directly

               self.run_step(batch, cond)  # Pass both batch and cond to run_step

               # Your existing logging and saving logic
               if self.step % self.log_interval == 0:
                   logger.dumpkvs()
               if self.step % self.save_interval == 0:
                   self.save()
                   # Additional break condition for testing
                   if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                       return
               self.step += 1

           # Save the last checkpoint if it wasn't already saved.
           if (self.step - 1) % self.save_interval != 0:
               self.save()


    def run_step(self, batch, cond):
        # Zero out the gradients for the generator
        self.opt.zero_grad()
        self.opt_D.zero_grad()  # Assuming opt_D is the discriminator's optimizer

        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        # Initialize GAN loss component for each forward-backward pass
        gan_loss_component = 0
        generator_condition_met = False  # Indicates if any images in the batch meet the t < 400 criterion
        # Zero gradients for both generator and discriminator at the start
        self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {k: v[i: i + self.microbatch].to(dist_util.dev()) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}

            temp_filenames = micro_cond.get("image_filenames", ["image_{}".format(i) for i in range(len(micro))])
            if not temp_filenames:
                temp_filenames = ["image_{}".format(i) for i in range(len(batch))]
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            #print(f"Sampled t values: {t}")

            # Process real images for discriminator
            real_images = batch.to(dist_util.dev())
            real_preds = self.discriminator(real_images)
            real_loss = self.gan_loss(real_preds, target_is_real=True)

            # Generate fake images
            final_images = []  # Initialize a list to store final generated images
            noise_masks = torch.zeros_like(micro)
            for j, img_name in enumerate(temp_filenames):
                bb = self.load_bb_data(img_name)
                #noise_masks[j, :, bb['y']:bb['y'] + bb['h'], bb['x']:bb['x'] + bb['w']] = 1
                # Calculate the actual width and height based on your definitions
                actual_width = bb['w'] - bb['x']
                actual_height = bb['h'] - bb['y']
                # Apply the mask within the correct BB area
                noise_masks[j, :, bb['y']:bb['y'] + actual_height, bb['x']:bb['x'] + actual_width] = 1

            noise = torch.randn_like(micro) * noise_masks
            micro_noised = micro + noise
            micro_cond.pop("image_filenames", None)
            model_output = self.ddp_model(micro_noised, t, **micro_cond)
            
            # Prepare images for discriminator training
            discriminator_images = model_output[t < 200]  # Select images where t < 400
            #print(f"Step {self.step + self.resume_step}: Number of suitable images (t < 400): {discriminator_images.size(0)}")

            if discriminator_images.size(0) > 0:
                generator_condition_met = True  # Set the flag since we have suitable images
                real_images = batch.to(dist_util.dev())
                real_preds = self.discriminator(real_images)
                real_loss = self.gan_loss(real_preds, target_is_real=True)
                fake_preds = self.discriminator(discriminator_images.detach())
                fake_loss = self.gan_loss(fake_preds, target_is_real=False)
                d_loss = real_loss + fake_loss
                gan_loss_component = d_loss.item()  # Update the GAN loss component
                d_loss.backward()
                self.opt_D.step()
                #print(f"Discriminator Loss (d_loss): {d_loss.item()}")
                logger.logkv("d_loss", d_loss.item())
                
                # Saving images presented to the discriminator
                if (self.step + self.resume_step) % 1000 == 0:
                    save_dir = "/content/guided-diffusion/discriminator_images"
                    for j, image in enumerate(discriminator_images):
                        if j >= 2000:  # Limit the number of images to save
                            break
                        # Directly denormalize the image from [-1, 1] to [0, 1] for saving
                        image_to_save = (image + 1) / 2
                        image_to_save = image_to_save.clamp(0, 1)  # Ensure values are within [0, 1]
                        # Construct the file path
                         # Construct the file path using the original image filename
                        original_filename = temp_filenames[j].replace(".jpg", "").replace(".png", "")  # Assuming filenames have standard image extensions
                        image_save_path = os.path.join(save_dir, f"{original_filename}_step_{self.step+self.resume_step}.png")
                        #image_save_path = os.path.join(save_dir, f"generated_image_{self.step + self.resume_step}_{j}.png")
                        # Save the image
                        torchvision.utils.save_image(image_to_save, image_save_path)
            else:
                # No suitable images for this step, ensure gan_loss_component remains 0
                gan_loss_component = 0

            # Generator updates
            if generator_condition_met:
                fake_preds_for_gen = self.discriminator(model_output)
                gen_loss = self.gan_loss(fake_preds_for_gen, target_is_real=True)
            else:
                #gen_loss = 0  # No generator loss if condition is not met
                gen_loss = torch.tensor(0.0, device=batch.device, dtype=torch.float32) 

            # Calculate L1 Loss for generator 
            losses = self.diffusion.training_losses(self.ddp_model, micro, t, model_kwargs=micro_cond)
            original_diffusion_loss = (losses["loss"] * weights).mean()
            # Assuming real_images and model_output are properly defined
            l1_loss = F.l1_loss(model_output, real_images, reduction='mean')
            # Hyperparameter for scaling the L1 loss component
            lambda_l1 = 0.001  # This value is adjustable
            total_gen_loss = original_diffusion_loss + 0.001 * gen_loss + 0.001 * gan_loss_component + lambda_l1 * l1_loss
            self.mp_trainer.backward(total_gen_loss / weights.mean())
            #print(f"Generator Loss (gen_loss): {gen_loss.item()}")
            #print(f"L1_loss (l1_loss): {l1_loss.item()}")
            logger.logkv("gen_loss", gen_loss.item())
            logger.logkv("l1_loss", l1_loss.item())

            #print(f"Total Generator Loss (total_gen_loss): {total_gen_loss.item()}")
            logger.logkv("total_gen_loss", total_gen_loss.item())

            # Handle non-syncing for DDP
            compute_losses = functools.partial(self.diffusion.training_losses, self.ddp_model, micro, t, model_kwargs=micro_cond)
            if 'image_filenames' in micro_cond:
                del micro_cond['image_filenames']
            
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())
            
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})



    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return "/content/guided-diffusion/checkpoints"
    #return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

# Directory for saving images
save_dir = "/content/guided-diffusion/images_noised"
os.makedirs(save_dir, exist_ok=True)


def save_noised_images_with_bb(images, noise_masks, filenames, step, load_bb_data_method, save_dir="/fred/oz097/latzi/FocusedDiffusionGAN_L1/guided-diffusion/images_noised"):
    #print("save_noised_images_with_bb function called.") 
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, mask) in enumerate(zip(images, noise_masks)):
        img_name = filenames[i]
        bb = load_bb_data_method(img_name)  # Use the passed method to load bounding box data
        noise = torch.randn_like(img) * mask
        noised_image = img + noise
        noised_image = noised_image.clamp(0, 1)

        # Highlight the bounding box
        # Ensure you are using the correct dictionary keys for the bounding box
        noised_image[:, bb['y']:bb['y']+1, bb['x']:bb['x']+bb['w']] = 1  # Top border
        noised_image[:, bb['y']+bb['h']-1:bb['y']+bb['h'], bb['x']:bb['x']+bb['w']] = 1  # Bottom border
        noised_image[:, bb['y']:bb['y']+bb['h'], bb['x']:bb['x']+1] = 1  # Left border
        noised_image[:, bb['y']:bb['y']+bb['h'], bb['x']+bb['w']-1:bb['x']+bb['w']] = 1  # Right border

        save_path = os.path.join(save_dir, f"noised_bb_image_{i}_step_{step}.png")
        vutils.save_image(noised_image, save_path)
