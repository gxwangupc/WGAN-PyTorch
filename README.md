# Wasserstein GAN (WGAN)
-------------------------------------------------
## Introduction
 * I have adapted the original author's code:<br>
<https://github.com/martinarjovsky/WassersteinGAN> <br>
 * The file download_lsun.py comes from a nice repository for downloading LSUN dataset:<br>
<https://github.com/fyu/lsun> <br>
 * I have added massive comments for the code. Hope it beneficial for understanding the WGAN, especially for a beginner.

## Environment & Requirements
* CentOS Linux release 7.2.1511 (Core)<br>
* python 3.6.5<br>
* pytorch  1.0.0<br>
* torchvision<br>
* argparse<br>
* os<br>
* random<br>
* json<br>
* subprocess<br>
* urllib

## Usage
### Train WGAN with cifar10:<br>

    python3 main.py --dataset cifar10 --cuda
Two folders will be created, i.e., `./data` & `./results`. The `./data` folder stores dataset. <br>
The `./results` folder contains two subfolders to store the generated samples and the trained models.<br> 
Training with lsun is also available.
### Download lsun dataset:<br>

    python3 download_lsun.py --category bedroom 
Download data for bedroom and save it to ./data.<br>
By replacing the option of `./--category`, you can download data of each category in LSUN as well.<br>
    ```
    python3 download_lsun.py 
    ```
    <br>
Download the whole data set.<br> 
### Generate images using the trained model: <br>

    python3 GenerateImg.py --config ./results/models/generator_config.json --weights ./results/models/netG_epoch_24.pth --output ./output --nimgs 100 --cuda

You can replace the above options as you want.

## NOTE
 * I have treated the WGAN in terms of DCGAN as default.
 * You can test the WGAN without batch normalizaiton by adding an option '--noBN'.<br>
 * You can also test WGAN in terms of MLP by adding an option '--MLP'.<br>
 * CPU is supported but training is very slow. You can run the code without the option'--cuda'.<br> 
 
## References 
1. <https://github.com/martinarjovsky/WassersteinGAN> <br>
2. <https://github.com/fyu/lsun> <br>
