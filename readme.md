# [*Diffusion Models as Masked Autoencoders (ICCV 2023)*](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Diffusion_Models_as_Masked_Autoencoders_ICCV_2023_paper.pdf)


## Installation
`pip install -r requirements.txt`

## Dataset (smithsonian_butterflies_subset)
https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset

## Run
`python main_diffmae.py`

## Results
See logs by `tensorboard --logdir logs`.

Pretrained models are in [Google drive](https://drive.google.com/file/d/12-sM0zY7VCHc1B40PZ-728OJmxkSrQZF/view?usp=drive_link).

Visualization of the first 4 images on 'smithsonian_butterflies_subset' dataset:

![avatar](fig/butterfly_results.png)