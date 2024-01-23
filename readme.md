# Implementation of [*Diffusion Models as Masked Autoencoders*](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Diffusion_Models_as_Masked_Autoencoders_ICCV_2023_paper.pdf).


### Installation
`pip install -r requirements.txt`

### Dataset (smithsonian_butterflies_subset)
https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset

### Run
```bash
# pretrained with diffmae
python mae_diffmae.py
```

### Results
See logs by `tensorboard --logdir logs`.

Weights are in [github release](https://github.com/IcarusWizard/MAE/releases/tag/cifar10).

Tensorboard logs are in [tensorboard.dev](https://tensorboard.dev/experiment/zngzZ89bTpyM1B2zVrD7Yw/#scalars).

Visualization of the first 4 images on 'smithsonian_butterflies_subset' validation dataset:

![avatar](pic/mae-cifar10-reconstruction.png)
