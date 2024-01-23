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
from utils import setup_seed

from transformers import ViTMAEForPreTraining
from transformers import ViTFeatureExtractor
from transformers import ViTMAEModel, ViTMAEConfig
import cv2
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--gpu', type=str, default='5')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    setup_seed(args.seed)

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

    # train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    # val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    # dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain2'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    config = ViTMAEConfig(image_size=args.image_size, patch_size=16, mask_ratio=0.25)
    # model = ViTMAEModel(config=config).to(device)
    # model = ViTMAEForPreTraining(config=config).to(device)
    # model = model.from_pretrained("facebook/vit-mae-base").to(device)
    # model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(device)
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=config).to(device)
    # feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    # optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    # lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        bar = tqdm(iter(dataloader), desc='Training')
        for batch in bar:
            step_count += 1
            img = batch['images'].to(device)

            outputs = model(img)
            y = model.unpatchify(outputs.logits)

            mask = outputs.mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
            mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

            loss = outputs.loss

            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            bar.set_description('loss: %.08f' % (avg_loss))

        writer.add_scalar('mae_loss', avg_loss, global_step=e)

        ''' visualize the first 4 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            img = torch.stack([dataset[i]['images'].to(device) for i in range(4)])
            # img = batch['images'].to(device)

            outputs = model(img)
            y = model.unpatchify(outputs.logits)
            # y = torch.einsum('nchw->nhwc', y).detach().cpu()

            mask = outputs.mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
            mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
            # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

            # output = (y+1).detach().cpu().numpy() * 127.5
            # for i in range(len(img)):
                # cv2.imwrite('results/%04d_%04d.png' % (e, i), output[i])
            writer.add_images('mae_image', (y.detach().cpu() + 1) / 2, global_step=e)
            writer.add_images('gt_image', (img.detach().cpu() + 1) / 2, global_step=e)
            writer.add_images('input_image', ((img * mask).detach().cpu() + 1) / 2, global_step=e)
        
        ''' save model '''
        torch.save(model, args.model_path)