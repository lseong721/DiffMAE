# [*Diffusion Models as Masked Autoencoders (ICCV 2023)*](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Diffusion_Models_as_Masked_Autoencoders_ICCV_2023_paper.pdf).


## Installation
`pip install -r requirements.txt`

## Dataset (smithsonian_butterflies_subset)
https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset

## Run
```bash
# pretrained with diffmae
python mae_diffmae.py
```

## Results
See logs by `tensorboard --logdir logs`.

Pretrained models are in [Google drive](https://github.com/IcarusWizard/MAE/releases/tag/cifar10).

Visualization of the first 4 images on 'smithsonian_butterflies_subset' dataset:

![avatar](fig/butterfly_results.png)
