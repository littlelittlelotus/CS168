import torch
import shutil
import torch.optim as optim
import time
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import utils

import json

def get_dataset_parts(image_dir, log_file, num_source=3, tr_ratio=0.85, val_ratio=0.15):
	"""
	----------------------------------------------------------------------------------------------------------
	Get patient names which has at least num_source+1 (e.g. 4) measurements. 
	We will use first num_source measurements (starting from the first measurment) as input and the following measurement as target,
	and we'll do this for all measurments (starting from the second measurement and so on)	 
	"""

	files = os.listdir(image_dir)
	all_patients = [f for f in files if f[0] == 'A' and not '.zip' in f]
	patients = [] # patients to be used for dataset
	for p in all_patients:
		pat_dir = os.path.join(image_dir, p, )
		subdir = os.listdir(pat_dir)
		subdir_files = [x for x in subdir if x[0] == '2']
		if len(subdir_files) >= num_source+1:
			patients.append(p)

	utils.print_and_save_msg(f'\n\n\nPatient ID\'s to be used for training the model (in total {len(patients)} patients):\n{patients}\n\n\n', log_file)

	# partition the dataset into train/val/test splits
	np.random.shuffle(patients)
	len_train = int(len(patients)*tr_ratio)
	len_valid = int(len(patients)*val_ratio)
	train_patients = patients[:len_train]
	valid_patients = patients[len_train:len_train+len_valid]
	test_patients = patients[len_train+len_valid:]

	utils.print_and_save_msg(f'\n\nTrain patients: {train_patients}\n\nValidation Patients: {valid_patients}\n\nTest Patients: {test_patients}\n\n', log_file)
	return train_patients, valid_patients, test_patients


# utils.save_model(epoch, opt, valid_im_loss, netG, optimizer_G, netD, optimizer_D, Conv3D, gen_loss, dis_loss)
def save_model(epoch, opt, im_loss, netG, optimizer_G, netD, optimizer_D, Conv3D, gen_loss, dis_loss):

	# Save the model with state dictionary:
	state = {'epoch': epoch, \
			 'opt': opt, \
			 'Generator': netG.state_dict(), \
			 'optimizer_G':optimizer_G.state_dict(), \
			 'Discriminator' : netD.state_dict(), \
			 'optimizer_D':optimizer_D.state_dict(), \
			 'Conv3D' : Conv3D.state_dict(), \
			 'Gen_Loss' : gen_loss, \
			 'Dis_Loss' : dis_loss, \
			 'im_loss': im_loss, \
			 'best_im_loss': opt.min_loss
			}

	chck_path = os.path.join(opt.chkpnt_dir, f'model_last.pth.tar')
	torch.save(state, chck_path)

	if im_loss < opt.min_loss:
		opt.min_loss = im_loss
		shutil.copyfile(chck_path, os.path.join(opt.chkpnt_dir, f'model_best.pth.tar'))

def get_function(func):

	if func == 'mse_loss':
		return torch.nn.MSELoss().cuda()
	elif func == 'l1_loss':
		return torch.nn.L1Loss().cuda()
	elif func == 'bce_loss':
		return torch.nn.BCELoss().cuda()

def get_optimizer(func, params, lr, opt):
	
	if func == 'adam':
		return optim.Adam(params, lr=lr, betas=(opt.beta1, 0.999))
	elif func == 'sgd':
		return optim.SGD(params, lr=lr)
	

# set requies_grad=Fasle to avoid computation
def set_requires_grad(nets, requires_grad=False):
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad



def print_and_save_msg(msg, FILE):
	print(msg)
	with open(FILE, 'a') as f:
		f.write(f'\n{msg}')


def save_images(real_1, fake_1, name='', ext='.png',epoch=0, im_path='./', mode='', cmap=None):

	fig=plt.figure(figsize=(15,8))
	if real_1.shape[-1] == 1:
		real_1 = np.squeeze(real_1, -1)
		fake_1 = np.squeeze(fake_1, -1)
		cmap = 'gray'		

	real_1 = real_1*0.5+0.5
	fake_1 = fake_1*0.5+0.5

	ax=plt.subplot(121)
	ax.set_title('real target image', fontsize=10)
	plt.imshow(real_1, cmap=cmap) 

	ax=plt.subplot(122)
	ax.set_title('fake interpolated image', fontsize=10)
	plt.imshow(np.clip(fake_1, 0.0, 1.0), cmap=cmap)

	if mode == 'test':
		img_plot_name = f'{im_path}/{name}{ext}'
	else:
		img_plot_name = f'{im_path}/epoch_{epoch:02d}_{mode}{ext}'

	plt.savefig(img_plot_name)
	plt.close(fig)
	plt.clf()

	## individual files
	fig=plt.figure()
	plt.imshow(real_1, cmap=cmap) 	
	plt.savefig(f'{im_path}/epoch_{epoch:02d}_{mode}_real{ext}')
	plt.close(fig)
	plt.clf()

	fig=plt.figure()
	plt.imshow(fake_1, cmap=cmap) 
	plt.savefig(f'{im_path}/epoch_{epoch:02d}_{mode}_fake{ext}')
	plt.close(fig)
	plt.clf()


def update_loss_graph(train_im_loss_list, valid_im_loss_list, gen_loss_list, dis_loss_list, exp_dir='./'):

	# if epoch % 10 == 0:
	plt.figure(figsize=(20,10))

	plt.subplot(121)
	plt.plot(np.arange(0,len(train_im_loss_list)), train_im_loss_list, label='Train Image Loss')
	plt.plot(np.arange(0,len(valid_im_loss_list)), valid_im_loss_list, label='Valid Image Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(122)
	plt.plot(np.arange(0,len(gen_loss_list)), gen_loss_list, label='Generator Loss')
	plt.plot(np.arange(0,len(dis_loss_list)), dis_loss_list, label='Discriminator Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(f'{exp_dir}/losses_over_epochs.png')
	plt.close()


def convert_to_numpy(tensr):

	nump = tensr.detach().cpu().numpy()
	nump = np.swapaxes(nump, 0, 2)
	nump = np.swapaxes(nump, 0, 1)
	return nump#*0.5+0.5