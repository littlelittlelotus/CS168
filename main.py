import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import numpy as np 

import options 
import get_data 
import utils
import networks 
import training

opt = options.training_options()
np.random.seed(1234) ## don't change this number in order for that the data partitions should be same for different running of the code.


train_patients, valid_patients, test_patients = utils.get_dataset_parts(opt.image_dir, opt.log_file, opt.num_source, opt.train_ratio, opt.valid_ratio)

# get the datasets ready for iterations
train_dataset = get_data.DatasetFromFolder(opt.image_dir, train_patients, opt.num_cross, opt.num_source, \
										   opt.input_ch, opt.output_ch, opt.log_file, opt.height_image, transform=transforms.Compose([
																									transforms.ToTensor()
																									# , normalize
																									]))

valid_dataset = get_data.DatasetFromFolder(opt.image_dir, valid_patients, opt.num_cross, opt.num_source, \
											opt.input_ch, opt.output_ch, opt.log_file, opt.height_image, transform=transforms.Compose([
																									transforms.ToTensor()
																									# , normalize
																									]))

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.workers)
valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, num_workers=opt.workers)

print('length of train dataset: ', len(train_dataset))
print('length of valid dataset: ', len(valid_dataset))

netG = networks.Generator(opt).cuda()
print('\nGenerator:\n', netG)
netD = networks.NLayerDiscriminator(opt).cuda()
print('\nDiscriminator:\n', netD)
Conv3D = networks.Conv3DBlock(opt).cuda()
print('\nConv3D blocks:\n', Conv3D)

optimizer_G = utils.get_optimizer(opt.optim_G, list(netG.parameters()) + list(Conv3D.parameters()), opt.lr_gen, opt)
optimizer_D = utils.get_optimizer(opt.optim_D, netD.parameters(), opt.lr_dis, opt)

# resume to an old training or start from scratch:
if opt.resume_dir:
	print('\n\n\nResuming to old training...\n\n\n')
	model_path = os.path.join(opt.chkpnt_dir, 'model_last.pth.tar')

	if os.path.isfile(model_path):
		utils.print_and_save_msg(f"=> loading checkpoint '{model_path}'", opt.log_file)
		checkpoint = torch.load(model_path) 

		netG.load_state_dict(checkpoint['Generator'])
		netD.load_state_dict(checkpoint['Discriminator'])
		Conv3D.load_state_dict(checkpoint['Conv3D'])

		optimizer_G.load_state_dict(checkpoint['optimizer_G'])
		optimizer_D.load_state_dict(checkpoint['optimizer_D'])

		## upload options:
		opt = checkpoint['opt']
		st_epoch = checkpoint['epoch'] + 1 # new starting epoch number
		opt.min_loss = checkpoint['best_im_loss']

		# print('\n\n\nopt-batch_size: ', opt.batch_size)
		# exit()
				
		utils.print_and_save_msg(f"=> loaded model checkpoint '{model_path}' that is trained for {st_epoch-1} epochs. (training will start from epoch {st_epoch})", opt.log_file)
	else:
		print(f"=> no checkpoint found at '{model_path}', exiting from the program...")
else:
	st_epoch = 1
	utils.print_and_save_msg('\n\n\nStarting to a new training from scratch..\n\n', opt.log_file)


for epoch in range(st_epoch, opt.max_epoch+1):
	
	_, _, train_im_loss = training.propagate(train_loader, epoch, netG, netD, Conv3D, optimizer_G, optimizer_D, opt, mode='train')
	with torch.no_grad():
		gen_loss, dis_loss, valid_im_loss = training.propagate(valid_loader, epoch, netG, netD, Conv3D, optimizer_G, optimizer_D, opt, mode='valid')	

	# plot loss function:
	opt.train_im_loss_list.append(train_im_loss)	
	opt.valid_im_loss_list.append(valid_im_loss)
	opt.gen_loss_list.append(gen_loss)
	opt.dis_loss_list.append(dis_loss)

	utils.update_loss_graph(opt.train_im_loss_list, opt.valid_im_loss_list, opt.gen_loss_list, opt.dis_loss_list, exp_dir=opt.exp_dir)
	utils.save_model(epoch, opt, valid_im_loss, netG, optimizer_G, netD, optimizer_D, Conv3D, gen_loss, dis_loss)





