import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'

import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torchvision import transforms
from tqdm import tqdm

# from model import *
from utils import setup_seed

from transformers import ViTMAEForPreTraining, ViTDiffMAEForPreTraining
from transformers import ViTFeatureExtractor
from transformers import ViTMAEModel, ViTMAEConfig, ViTDiffMAEConfig
import cv2
from datasets import load_dataset
from accelerate import Accelerator
import numpy as np
# from diffusers import DDPMPipeline
from diffusers import DDPMPipeline, DDPMScheduler

def main(args):
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}


    dataset.set_transform(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=load_batch_size, shuffle=True)

    accelerator = Accelerator(log_with='tensorboard', 
                              project_dir='logs')

    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain2'))
    device = accelerator.device

    config = ViTDiffMAEConfig(image_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
    model = ViTDiffMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=config)#.to(device)
    # config = ViTMAEConfig(image_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
    # model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=config)#.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=[0.9, 0.95], weight_decay=args.weight_decay)

    model, optim, dataloader = accelerator.prepare(model, optim, dataloader)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    global_step = 0
    optim.zero_grad()
    for epoch in range(args.total_epoch):
        model.train()
        losses = []

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            global_step += 1
            img_clean = batch['images'].to(device)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, [len(img_clean),], device=device).long()
            noise = torch.randn(img_clean.shape).to(img_clean.device)
            img_noise = noise_scheduler.add_noise(img_clean, noise, timesteps)

            with accelerator.accumulate(model):

                outputs = model(img_clean, img_noise)
                # outputs = model(img_clean)

                patch_input = model.patchify(img_clean)
                patch_pred = outputs.logits
                patch_mask = outputs.mask

                loss = (patch_pred - patch_input) ** 2
                loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
                loss = (loss * patch_mask).sum() / patch_mask.sum()  # mean loss on removed patches
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()


            losses.append(loss.detach().item())

            # writer.add_scalar('mae_loss', loss.detach().item(), global_step=global_step)
            writer.add_scalar('mae_loss', np.array(losses).mean(), global_step=global_step)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
        progress_bar.close()

        accelerator.wait_for_everyone()

        ''' visualize the first 4 predicted images on val dataset'''
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                img_clean = torch.stack([dataset[i]['images'].to(device) for i in range(4)])
                timesteps = torch.full([len(img_clean),], noise_scheduler.config.num_train_timesteps-1, device=device).long()
                noise = torch.randn(img_clean.shape).to(device)
                img_noise = noise_scheduler.add_noise(img_clean, noise, timesteps)

                for idx, t in tqdm(enumerate(noise_scheduler.timesteps)):
                    if idx == 0:
                        outputs = model(img_clean, img_noise)
                    else:
                        outputs = model(img_clean, img_noise, noise=noise_seed)

                    # 1. predict noise model_output
                    img_pred = outputs.logits.detach()
                    img_pred = model.unpatchify(img_pred)

                    mask = outputs.mask.detach()
                    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
                    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

                    noise_seed = outputs.noise.detach()

                    # 2. compute previous image: x_t -> x_t-1
                    img_noise = noise_scheduler.add_noise(img_clean, img_noise, t)
                    # generator = torch.Generator(device=device).manual_seed(0)
                    # img_noise = noise_scheduler.step(img_pred, t, img_noise, generator=generator).prev_sample

                    # img_save = img_noise[0].permute(1, 2, 0).detach().cpu().numpy()
                    # cv2.imwrite('results/%04d.png' % t, (img_save + 1) * 127.5)

                mask = outputs.mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
                mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

                writer.add_images('mae_image', (img_pred.detach().cpu() + 1) / 2, epoch)
                writer.add_images('input_image', ((img_clean * mask + noise * (1 - mask)).detach().cpu() + 1) / 2, epoch)
            
        ''' save model '''
        torch.save(model, args.model_path)
    accelerator.end_training()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--gpu', type=str, default='5,6,7')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")

    args = parser.parse_args()
    setup_seed(args.seed)

    main(args)