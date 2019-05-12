import torch
import argparse
import torchvision
import os
import json

from models.DCGAN_G import DCGAN_G, DCGAN_G_nobn
from models.MLP_G import MLP_G

parser = argparse.ArgumentParser()
parser.add_argument('--config',  required=True, type=str, help='path to generator config .json file')
parser.add_argument('--weights', required=True, type=str, help='path to generator weights .pth file')
parser.add_argument('--output', required=True, type=str, help="path to to output directory")
parser.add_argument('--nimgs', required=True, type=int, help="number of images to generate", default=1)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()

if not os.path.exists(opt.output):
    os.mkdir(opt.output)
"""
Use GPUs if available.
"""
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")

"""
Load config.
"""
with open(opt.config, 'r') as gencfg:
    generator_config = json.loads(gencfg.read())

nz = generator_config["nz"]
nc = generator_config["nc"]
ngf = generator_config["ngf"]
isize = generator_config["isize"]
ngpu = generator_config["ngpu"]
noBN = generator_config["noBN"]
MLP_G = generator_config["MLP_G"]

if noBN:
    netG = DCGAN_G_nobn(nz, ngf, nc, ngpu)
elif MLP_G:
    netG = MLP_G(nz, ngf, nc, isize, ngpu)
else:
    netG = DCGAN_G(nz, ngf, nc, ngpu)

"""
Load weights
"""
netG.load_state_dict(torch.load(opt.weights))

"""
Initialize noise
"""
if opt.cuda:
    netG = netG.to(device)
    fixed_noise = torch.randn(opt.nimgs, nz, 1, 1, device=device)
else:
    fixed_noise = torch.randn(opt.nimgs, nz, 1, 1)

fake = netG(fixed_noise)

for i in range(opt.nimgs):
    torchvision.utils.save_image(fake.data[i, ...].reshape((1, nc, isize, isize)),
                                 os.path.join(opt.output, "generated_%02d.png" % i),
                                 normalize=True)
