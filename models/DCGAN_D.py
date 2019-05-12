import torch

"""
The discriminator network in terms of DCGAN.
    nc  : number of color channels
    ndf : size of feature maps of D, 64 default
    ngpu: number of CUDA devices available
"""

class DCGAN_D(torch.nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DCGAN_D, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        '''
        torch.nn.Conv2d: Applies a 2D convolution operator over an input image.
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,\
         bias=True, padding_mode='zeros')
        '''
        self.main = torch.nn.Sequential(
            #3->64, stride=2, padding=1
            torch.nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            ##If you need extra layers, uncomment this block and copy it the number you want.
            ##64->64, kernel_size=1, stride=1, padding=1
            # torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            # torch.nn.BatchNorm2d(64),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            #64->128, stride=2, padding=1
            torch.nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #128->256, stride=2, padding=1
            torch.nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #256->512, stride=2, padding=1
            torch.nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #512->1, stride=1, padding=0
            torch.nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
        )

    """
    Forward propogation of D.
    """
    '''
    #torch.nn.parallel.data_parallel(module, inputs, device_ids=None, \
     output_device=None, dim=0, module_kwargs=None)
     #module: the module to evaluate in parallel, self.net
     #input : inputs to the module
     #device_ids:GPU ids on which to replicate module
     #output_device:GPU location of the output Use -1 to indicate the CPU. (default: device_ids[0])
    '''
    def forward(self, input):
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu >1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)


"""
The discriminator network in terms of DCGAN without batch normalization.
    nc  : number of color channels
    ndf : size of feature maps of D, 64 default
    ngpu: number of CUDA devices available
"""

class DCGAN_D_nobn(torch.nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DCGAN_D_nobn, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        '''
        torch.nn.Conv2d: Applies a 2D convolution operator over an input image.
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,\
         bias=True, padding_mode='zeros')
        '''
        self.main = torch.nn.Sequential(
            #3->64, stride=2, padding=1
            torch.nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            ##If you need extra layers, uncomment this block and copy it the number you want.
            ##64->64, kernel_size=1, stride=1, padding=1
            # torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            #64->128, stride=2, padding=1
            torch.nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #128->256, stride=2, padding=1
            torch.nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #256->512, stride=2, padding=1
            torch.nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #512->1, stride=1, padding=0
            torch.nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
        )

    """
    Forward propogation of D.
    """
    '''
    #torch.nn.parallel.data_parallel(module, inputs, device_ids=None, \
     output_device=None, dim=0, module_kwargs=None)
     #module: the module to evaluate in parallel, self.net
     #input : inputs to the module
     #device_ids:GPU ids on which to replicate module
     #output_device:GPU location of the output Use -1 to indicate the CPU. (default: device_ids[0])
    '''
    def forward(self, input):
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu >1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)

