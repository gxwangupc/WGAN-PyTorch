import torch

"""
The generator network in terms of MultiLayer Perception.
    nz   : input latent vector
    ngf  : size of feature maps of G, 64 default  
    nc   : number of color channels
    isize: image size
    ngpu : number of CUDA devices available
"""
class MLP_G(torch.nn.Module):
    def __init__(self, nz, ngf, nc, isize, ngpu):
        super(MLP_G, self).__init__()
        self.nz = nz
        self.ngf =ngf
        self.nc = nc
        self.isize = isize
        self.ngpu = ngpu

        self.main = torch.nn.Sequential(
            #100->64
            torch.nn.Linear(self.nz, self.ngf),
            torch.nn.ReLU(inplace=True),
            #64->64
            torch.nn.Linear(self.ngf, self.ngf),
            torch.nn.ReLU(inplace=True),
            #64->64
            torch.nn.Linear(self.ngf, self.ngf),
            torch.nn.ReLU(inplace=True),
            #64->3*64*64
            torch.nn.Linear(self.ngf, self.nc * self.isize * self.isize),
        )

    """
    Forward propogation of G.
    """
    '''
    #torch.nn.parallel.data_parallel(module, inputs, device_ids=None, \
     #output_device=None, dim=0, module_kwargs=None)
     #module: the module to evaluate in parallel, self.net
     #input : inputs to the module
     #device_ids:GPU ids on which to replicate module
     #output_device:GPU location of the output Use -1 to indicate the CPU. (default: device_ids[0])
    '''
    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)
