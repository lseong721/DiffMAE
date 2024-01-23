import torch

img_clean = torch.load('results/img_clean_nor_000.pt')
img_pred = torch.load('results/img_pred_nor_000.pt')

print(img_clean-img_pred)
print(img_clean.shape)
print(img_pred.shape)
