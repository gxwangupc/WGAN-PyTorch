import torch
import argparse
import os
import torchvision
import random
import json

from models.DCGAN_G import DCGAN_G, DCGAN_G_nobn
from models.DCGAN_D import DCGAN_D, DCGAN_D_nobn
from models.MLP_G import MLP_G
from models.MLP_D import MLP_D



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default="./data", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in G')
parser.add_argument('--ndf', type=int, default=64, help='size of feature maps in D')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--MLP_G', action='store_true', help='use MLP for G')
parser.add_argument('--MLP_D', action='store_true', help='use MLP for D')
parser.add_argument('--results', default="./results", help='Where to store samples and models')
parser.add_argument('--samples', default="./results/samples", help='Where to store samples')
parser.add_argument('--models', default="./results/models", help='Where to store models')
parser.add_argument('--Adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()
print(opt)

"""
Path to save samples and models, respectively.
"""
os.makedirs("results/samples/", exist_ok=True)
os.makedirs("results/models/", exist_ok=True)

"""
Set random seed for reproducibility.
"""
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000) # fix seed# use if you want new results
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
#Sets the seed for generating random numbers. Returns a torch._C.Generator object.
torch.manual_seed(opt.manualSeed)

#This line may increase the training speed a bit.
torch.backends.cudnn.benchmark = True

"""
Use GPUs if available.
"""
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")

"""
Create the dataset.
"""
'''
#torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, \
 loader=<function default_loader>, is_valid_file=None)
 #root: Root directory path.
 #transform: transform (callable, optional): A function/transform that takes in \
   #an PIL image and returns a transformed version. 
'''
'''
torchvision.datasets: contains a lot of datasets
'''
'''
#torchvision.transforms.Compose(transforms): Composes several transforms together.
#torchvision.transforms.Resize(size, interpolation=2): Resize the input PIL Image to the given size.
 #interpolation: Desired interpolation. Default is PIL.Image.BILINEAR.
#torchvision.transforms.CenterCrop(size): Crops the given PIL Image at the center.\
#torchvision.transforms.ToTensor(): Convert a PIL Image or numpy.ndarray to tensor.
#torchvision.transforms.Normalize(mean, std, inplace=False):Normalize a tensor image with mean and standard deviation. \
 #Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input.
 #mean: Sequence of means for each channel.
 #std : Sequence of standard deviations for each channel.
'''
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = torchvision.datasets.ImageFolder(root=opt.dataroot,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(opt.img_size),
                                torchvision.transforms.CenterCrop(opt.img_size),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
elif opt.dataset == 'lsun':
    dataset = torchvision.datasets.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.Resize(opt.img_size),
                            torchvision.transforms.CenterCrop(opt.img_size),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = torchvision.datasets.CIFAR10(root=opt.dataroot, download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.Resize(opt.img_size),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
    )
assert dataset
'''
Create the dataloader.
#torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,\
 collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
 #batch_size: how many samples per batch to load.
 #shuffle: set to True to have the data reshuffled at every epoch.
 #num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
'''
dataset = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=int(opt.workers))

"""
Custom weights initialization called on netG and netD.
All model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02. 
Note We set bias=False in both Conv2d and ConvTranspose2d.
"""
def weights_init(input):
    classname = input.__class__.__name__
    if classname.find('Conv') != -1:
        input.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        input.weight.data.normal_(1.0, 0.02)
        input.bias.data.fill_(0)
'''
Write out generator config to generate images together wth training checkpoints (.pth)
'''
generator_config = {"nz": opt.nz, "ngf": opt.ngf, "nc": opt.nc, "isize": opt.img_size, "ngpu": opt.ngpu, "noBN": opt.noBN, "MLP_G": opt.MLP_G}
with open(os.path.join(opt.models, "generator_config.json"), 'w') as gcfg:
    gcfg.write(json.dumps(generator_config)+"\n")
    
'''
Create a generator and apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
'''
if opt.noBN:
    netG = DCGAN_G_nobn(opt.nz, opt.ngf, opt.nc, opt.ngpu)
elif opt.MLP_G:
    netG = MLP_G(opt.nz, opt.ngf, opt.nc, opt.img_size, opt.ngpu)
else:
    netG = DCGAN_G(opt.nz, opt.ngf, opt.nc, opt.ngpu)

netG.apply(weights_init)

'''
Load the trained netG to continue training if it exists.
'''
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
    print("Loaded the saved netG model...")
print(netG)

'''
Create a discriminator and apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
'''
if opt.MLP_D:
    netD = MLP_D(opt.img_size, opt.nc, opt.ndf, opt.ngpu)
else:
    netD = DCGAN_D(opt.nc, opt.ndf, opt.ngpu)
    
netD.apply(weights_init)

'''
Load the trained netD to continue training if it exists.
'''
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
    print("Loaded the saved netD model...")
print(netD)

fixed_noise = torch.FloatTensor(torch.randn(opt.batch_size, opt.nz, 1, 1))
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.to(device)
    netG.to(device)
    one, mone = one.to(device), mone.to(device)
    fixed_noise = fixed_noise.to(device)

"""
Setup optimizer.
"""
if opt.Adam:
    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr = opt.lrG)

"""
Training.
"""
gen_iters = 0
print("Starting Training Loop...")
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataset, 0):
        """
        (1) Update D network.
        """
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iters < 25 or gen_iters % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters

        j = 0
        while j < Diters and i < len(dataset):
            j += 1
            '''
            Clamp parameters to a cube.
            '''
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            '''
            Train with real batch.
            '''
            netD.zero_grad()
            # Format batch
            real_cpu = data[0]
            if opt.cuda:
                real_cpu = real_cpu.to(device)
            batch_size = real_cpu.size(0)
            errD_real = netD(real_cpu)
            errD_real.backward(one)

            '''
            Train with fake batch.
            '''
            # Sample batch of latent vectors.
            with torch.no_grad():# totally freeze netG
                if opt.cuda:
                    noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
                else:
                    noise = torch.randn(batch_size, opt.nz, 1, 1)
            fake = netG(noise)
            errD_fake = netD(fake.detach())
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        """
        (2) Update G network.
        """
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataset,
        # make sure we feed a full batch of noise
        errG = netD(fake)
        errG.backward(one)
        optimizerG.step()
        gen_iters += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.nepoch, i, len(dataset), gen_iters,
            errD.item(), errG.item(), errD_real.item(), errD_fake.item()))

        if gen_iters % 500 == 0:
            torchvision.utils.save_image(real_cpu,
                                         '%s/real_samples.png' % opt.samples,
                                         normalize=True)
            fake = netG(fixed_noise)
            torchvision.utils.save_image(fake.detach(),
                                         '%s/fake_samples_epoch_%03d.png' % (opt.samples, gen_iters),
                                         normalize=True)
    """
    Save the trained models.
    """
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.models, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.models, epoch))
