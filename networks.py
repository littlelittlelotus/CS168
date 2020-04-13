import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of the paper "https://arxiv.org/abs/1707.00067".
Firstly, the GAN network described in the Section 3 is implemented
"""
class Generator(nn.Module):
	def __init__(self, opt):
		super(Generator, self).__init__()

		self.down_blocks = nn.ModuleList()
		self.num_down = opt.num_down

		in_ch = opt.input_ch
		out_ch = opt.ngf

		for ii in range(self.num_down): 
			self.down_blocks.append(DownBlock(in_ch=in_ch, out_ch=out_ch, is_first = (ii == 0), is_last = (ii == self.num_down-1)))
			in_ch = out_ch
			out_ch = out_ch * 2

		bn_ch = opt.ngf * (2**(self.num_down-1))

		self.up_blocks = nn.ModuleList()
		in_ch = bn_ch
		out_ch = in_ch // 2
		for ii in range(self.num_down): 
			self.up_blocks.append(UpBlock(in_ch=in_ch, out_ch=out_ch,  is_first = (ii == 0), is_last = (ii == self.num_down-1)))
			if ii > 0:
				in_ch = in_ch // 2
			out_ch = out_ch // 2 

		self.last_layer = nn.ModuleList()
		self.last_layer.append(nn.ReLU(True))
		# self.last_layer.append(nn.ConvTranspose2d(out_ch*2, opt.output_ch, kernel_size=3, stride=1, padding=1))
		# self.last_layer.append(nn.Sigmoid())		
		self.last_layer.append(nn.Conv2d(out_ch*2, opt.output_ch, kernel_size=3, stride=1, padding=1))
		self.last_layer.append(nn.Tanh())
		
	def forward(self, x):
		skip_cons = {}
		lvl = 0
		for down_layer in self.down_blocks:
			#print('\n\ndown layer inp shape', x.shape)
			x = down_layer(x)
			#print('down layer out shape', x.shape)
			skip_cons[f'skip_{lvl}'] = x
			lvl += 1

		# last downsampled and fused layer is botleneck layer
		inp = skip_cons[f'skip_{lvl-1}'] 

		for ii in range(self.num_down):			
			layer = self.up_blocks[ii]
			#print(f'\nup layer-{ii} inp shape', inp.shape)
			out = layer(inp)
			#print('up layer out shape', out.shape)
			if ii < (self.num_down -1):
				skip = skip_cons[f'skip_{lvl-2-ii}']
				#print('skip connection shape', skip.shape)
				inp = torch.cat([out, skip], 1)

		for layer in self.last_layer:
			out = layer(out)
		
		return out


class Conv3DBlock(nn.Module):
	def __init__(self, opt):
		super(Conv3DBlock, self).__init__()
		
		self.in_ch = opt.input_ch
		self.out_ch = opt.input_ch
		self.num_source = opt.num_source

		relu = nn.ReLU(True)
		conv = nn.Conv3d(self.in_ch, self.out_ch, kernel_size=(2,3,3), padding=(0,1,1), bias=False)
		norm = nn.BatchNorm3d(self.out_ch)

		model = [conv, relu, norm]
		self.model = nn.Sequential(*model)

	def forward(self, x):
		# Convert 4D tensor (2D image) into 5D tensor (3D image)
		# print('\nConv3D Block input shape: ', x.shape)
		bs, ch, h, w = x.shape
		inp = torch.zeros([bs, self.in_ch, self.num_source, h, w]).cuda()
		for ii in range(self.num_source):
			inp[:,:,ii,:,:] = x[:, ii*self.in_ch:(ii+1)*self.in_ch,:,:]
		# print('\n4D inp shape: ', inp.shape)
		
		out = inp
		for _ in range(self.num_source-1):
			out = self.model(out)
			#print('\noutput shape', out.shape)
		return torch.squeeze(out, 2)
		 
class DownBlock(nn.Module):
	def __init__(self, in_ch, out_ch, is_first=False, is_last=False):
		super(DownBlock, self).__init__()

		downrelu = nn.LeakyReLU(0.2, True)
		downconv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
		# downconv2 = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=1, padding=1, bias=False)
		downnorm = nn.BatchNorm2d(out_ch)

		if is_first:
			model = [downconv]
		elif is_last:
			model = [downrelu, downconv]
		else:
			model = [downrelu, downconv, downnorm]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		# #print('\n\ndownsampling input', x.shape)
		out = self.model(x)
		# #print('\ndownsampling out', out.shape)
		return out


class UpBlock(nn.Module):
	def __init__(self, in_ch, out_ch,  is_first=False, is_last=False):
		super(UpBlock, self).__init__()

		uprelu = nn.ReLU(True)
		# # #print('\n\nunder convtranspose inch outch', in_ch, out_ch)
		upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)		
		upnorm = nn.BatchNorm2d(out_ch)

		
		if is_first:
			model = [uprelu, upconv, upnorm]
		elif is_last:
			model = [uprelu, upconv, upnorm]
		else:			
			model = [uprelu, upconv, upnorm, nn.Dropout(0.5)]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		# #print('\n\nupsampling input', x.shape)
		out = self.model(x)
		# #print('\nupsampling out', out.shape)
		return out



# class Discriminator(nn.Module):
# 	def __init__(self):
# 		super(Discriminator, self).__init__()

# 		


"""
from pix2pix paper:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, opt, norm_layer=nn.BatchNorm2d, use_sigmoid=True, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        

        kw = 4
        padw = 1
       	num_inp = (opt.num_source+1)

        sequence = [
            nn.Conv2d(opt.input_ch*num_inp, opt.ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, opt.n_layers):
        # #print('n layers', opt.n_layers)
        # for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(opt.ndf * nf_mult_prev, opt.ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(opt.ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**opt.n_layers, 8)
        sequence += [
            nn.Conv2d(opt.ndf * nf_mult_prev, opt.ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(opt.ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(opt.ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # #print('input shape', input.shape)
        # #print('\noutput shape', self.model(input).shape)
        # exit()
        return self.model(input)
