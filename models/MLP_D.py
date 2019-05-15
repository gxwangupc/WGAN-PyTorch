import torch

"""
The discriminator network interms of MultiLayer Perception.
    nc  : number of color channels
    ndf : size of feature maps of D, 64 default.
    ngpu: number of CUDA devices available
"""
class MLP_D(torch.nn.Module):
    def __init__(self, isize, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.nc = nc
        self.isize = isize
        self.ndf = ndf
        self.ngpu = ngpu

        self.main = torch.nn.Sequential(
            #3*64*64->64
            torch.nn.Linear(self.nc * self.isize * self.isize, self.ndf),
            torch.nn.ReLU(inplace=True),
            #64->64
            torch.nn.Linear(self.ndf, self.ndf),
            torch.nn.ReLU(inplace=True),
            #64->64
            torch.nn.Linear(self.ndf, self.ndf),
            torch.nn.ReLU(inplace=True),
            #64->1
            torch.nn.Linear(self.ndf, 1),
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
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu >1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)
