import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torchvision import transforms
from tqdm import tqdm

# from model import *
# from utils import setup_seed
import torch.nn as nn

from transformers import ViTMAEForPreTraining, ViTDiffMAEForPreTraining
from transformers import ViTFeatureExtractor
from transformers import ViTMAEModel, ViTMAEConfig, ViTDiffMAEConfig
import cv2
from datasets import load_dataset
# from accelerate import Accelerator
import numpy as np
# from diffusers import DDPMPipeline
from diffusers import DDPMPipeline, DDPMScheduler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from data_loader import get_dataloaders
import random

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def transform(examples):
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def main(args, dataloader, model, optim, lr_scheduler, device):
    writer = SummaryWriter(os.path.join(args.log_dir, args.data_name, args.log_name))
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    global_step = 0
    optim.zero_grad()
    for epoch in range(args.total_epoch):
        model.train()
        losses = []

        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            global_step += 1
            img_clean = batch['images'].to(device)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, [len(img_clean),], device=device).long()
            noise = torch.randn(img_clean.shape).to(device)
            img_noise = noise_scheduler.add_noise(img_clean, noise, timesteps)

            outputs = model(img_clean, img_noise, timesteps)

            # patch_input = model.patchify(img_clean)
            # patch_pred = outputs.logits
            # patch_mask = outputs.mask

            loss = outputs.loss
            
            loss.backward()
            optim.step()
            optim.zero_grad()


            losses.append(loss.detach().item())

            # writer.add_scalar('mae_loss', loss.detach().item(), global_step=global_step)
            writer.add_scalar('mae_loss', np.array(losses).mean(), global_step=global_step)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)

        lr_scheduler.step()
        progress_bar.close()

        ''' visualize the first 4 predicted images on val dataset'''
        if epoch % 10 == 0:
            ''' save model '''
            torch.save(model.state_dict(), os.path.join(args.model_dir, '{}.pth'.format(args.log_name)))
            model.eval()
            with torch.no_grad():
                img_clean = torch.stack([dataset[i]['images'].to(device) for i in range(4)])
                timesteps = torch.full([len(img_clean),], noise_scheduler.config.num_train_timesteps-1, device=device).long()
                noise = torch.randn(img_clean.shape).to(device)

                time_schedules = torch.cat([noise_scheduler.timesteps[0::100], torch.tensor(0)[None]]).to(device)
                # time_schedules = noise_scheduler.timesteps.to(device)
                for i in tqdm(range(len(time_schedules)-1)):
                    if i == 0:
                        img_noise = noise_scheduler.add_noise(img_clean, noise, time_schedules[i])

                        outputs = model(img_clean, img_noise, time_schedules[i])
                        noise_seed = outputs.noise.detach()
                    else:
                        # 2. compute previous image: x_t -> x_t-1
                        img_noise = noise_scheduler.add_noise(img_pred, img_noise, time_schedules[i+1])
                        # img_noise = img_clean * (1 - mask) + img_noise * (mask)

                        outputs = model(img_clean, img_noise, time_schedules[i], noise=noise_seed)

                    # 1. predict noise model_output
                    img_pred = outputs.logits.detach()
                    img_pred = model.unpatchify(img_pred)
                    # img_noise = img_noise.clamp(-1, 1)

                    # img_noise = noise_scheduler.step(img_pred, t, img_noise, generator=generator).prev_sample

                    mask = outputs.mask.detach()
                    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * model.config.num_channels)  # (N, H*W, p*p*3)
                    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

                    # img_save = img_pred[0].permute(1, 2, 0).detach().cpu().numpy()
                    # cv2.imwrite('results/%04d-1.png' % time_schedules[i].item(), (img_save + 1) * 127.5)

                # exit()

                writer.add_images('gt_image', (img_clean.detach().cpu() + 1) / 2, epoch)
                writer.add_images('mae_image', (img_pred.detach().cpu() + 1) / 2, epoch)
                writer.add_images('input_image', ((img_clean * (1 - mask) + noise * (mask)).detach().cpu() + 1) / 2, epoch)
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75) # ratio to be removed
    parser.add_argument('--total_epoch', type=int, default=10000)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_name", type=str, default="diffmae")
    parser.add_argument("--data_name", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default="/hdd6/seongmin/DB/Hair/optim_neural_textures")
    parser.add_argument('--warmup_epoch', type=int, default=200)

    args = parser.parse_args()

    setup_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.model_dir, exist_ok=True)

    ''' Dataset '''
    dataset = load_dataset("huggan/smithsonian_butterflies_subset")['train']
    # dataset = load_dataset("huggan/inat_butterflies_top10k")['train']
    # dataset = load_dataset("huggan/flowers-102-categories")['train']
    # dataset = load_dataset('huggan/CelebA-HQ')['train']
    dataset.set_transform(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    # dataset = get_dataloaders(args)
    # dataloader = dataset['train']

    ''' Network '''
    config = ViTDiffMAEConfig(num_channels=3, image_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
    model = ViTDiffMAEForPreTraining(config=config).to(device)
    # model = torch.load('models/test_butterfly_ddim.pt')
    # model.config.mask_ratio = 0.5
    # model = torch.load('models/test_butterfly_ddim.pt')
    model.load_state_dict(torch.load('models/test_butterfly_ddim2.pth'))

    # model = ViTDiffMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=config).to(device)
    # config = ViTMAEConfig(image_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
    # model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=config)#.to(device)
    lr = args.learning_rate# * args.batch_size / 256
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.95], weight_decay=args.weight_decay)

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    main(args, dataloader, model, optim, lr_scheduler, device)