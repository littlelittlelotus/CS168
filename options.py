import argparse
from datetime import datetime
import os
import utils 
import numpy as np 

# https://github.com/Po-Hsun-Su/pytorch-ssim
import pytorch_ssim

def training_options():

	parser = argparse.ArgumentParser(description='Galucoma Progression Project')

	parser.add_argument('--resume', action='store_true', help='if set, previously trained model from resume_dir is restored.')
	parser.add_argument('--resume_dir', type=str, default='', \
		            	help='if specified, old training is resumed and the best model under this directory is loaded')

	parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
	parser.add_argument('--batch_size', default=8, type=int, help='')
	parser.add_argument('--image_dir', default='../../../data/glaucoma_progression/Patients_Data_Folders', type=str, help='')
	parser.add_argument('--num_source', default=3, type=int, help='number of images to be used as source channels')
	parser.add_argument('--height_image', default=400, type=int, help='height of the cropped images')	

	parser.add_argument('--train_ratio', default=0.85, type=float, help='fraction of train images over whole dataset')
	parser.add_argument('--valid_ratio', default=0.15, type=float, help='fraction of valid images over whole dataset')
	parser.add_argument('--num_cross', default=61, type=int, help='number of cross sections')

	parser.add_argument('--input_ch', default=3, type=int, help='# channels of the input images')
	parser.add_argument('--output_ch', default=3, type=int, help='# channels of the output images')

	parser.add_argument('--max_epoch', default=50, type=int, help='# epochs for training')

	parser.add_argument('--im_coeff', default=50.0, type=float, help='coefficient of the image loss in generator loss calculation')
	parser.add_argument('--ssim_coeff', default=0.0, type=float, help='coefficient of the SSIM loss in generator loss calculation')

	parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in first conv layer')
	parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in first conv layer')
	parser.add_argument('--num_down', default=4, type=int, help='# downsampling layers in the generator')
	parser.add_argument('--n_layers', default=3, type=int, help='# layers in the discriminator')
	parser.add_argument('--lr_gen', default=5e-4, type=float, help='learning rate for netG optimizer')
	parser.add_argument('--lr_dis', default=5e-4, type=float, help='learning rate for netD optimizer')
	parser.add_argument('--beta1', type=float, default=0.5, help='momentum term if adam optimizer is used')

	parser.add_argument('--adv_loss', type=str, default='mse_loss', help='adversarial loss function, options: mse_loss, bce_loss')
	parser.add_argument('--img_loss', type=str, default='l1_loss', help='adversarial loss function, options: mse_loss, l1_loss')
	parser.add_argument('--optim_G', type=str, default='adam', help='optimizer for the Generator')
	parser.add_argument('--optim_D', type=str, default='sgd', help='optimizer for the Discriminator')

	
	opt = parser.parse_args()

	opt.adv_loss_f = utils.get_function(opt.adv_loss)
	opt.img_loss_f = utils.get_function(opt.img_loss)
	opt.ssim_f = pytorch_ssim.ssim

	opt.train_im_loss_list = []
	opt.valid_im_loss_list = []
	opt.gen_loss_list = []
	opt.dis_loss_list = []

	opt.min_loss = np.inf

	if opt.resume_dir:
		opt.exp_dir = opt.resume_dir
	else:
		opt.exp_dir = str(datetime.now()).replace(' ', '__').replace('-', '_').replace(':', '_').replace('.', '_')[:20] # YY_MM_DD___HH_MM_SS

	os.system(f"mkdir {opt.exp_dir}")
	opt.im_path = os.path.join(opt.exp_dir, 'images')
	os.system(f"mkdir {opt.im_path}")
	opt.chkpnt_dir = os.path.join(opt.exp_dir,'checkpoints')
	os.system(f"mkdir {opt.chkpnt_dir}")

	opt.log_file = f'{opt.exp_dir}/log_file.txt'


	cfg = ''
	cfg += '----------------- Options ---------------\n'
	for k, v in sorted(vars(opt).items()):
		cfg += f'{str(k):>25}: {str(v):<30}\n'
	cfg += '----------------- End -------------------'
	print(cfg)
	
	with open(f'{opt.exp_dir}/config.txt', 'w') as f:
		f.write(f'{cfg}')

	return opt

def test_options():

	parser = argparse.ArgumentParser(description='Galucoma Progression Project Testing')

	parser.add_argument('--exp_dir', type=str, default='', help='the directory of the experiment')

	parser.add_argument('--test_target', default='all', type=str, help='cross sections to be tested, options: all, single')	
	parser.add_argument('--cross_ID', default=31, type=int, help='cross section ID to be tested, used just when test_target is set to single')	

	parser.add_argument('--num_cross', default=61, type=int, help='number of cross sections')

	parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
	parser.add_argument('--batch_size', default=1, type=int, help='')
	parser.add_argument('--image_dir', default='../../../data/glaucoma_progression/Test_Patients', type=str, help='')
	parser.add_argument('--num_source', default=3, type=int, help='number of images to be used as source channels')
	parser.add_argument('--height_image', default=400, type=int, help='height of the cropped images')	

	parser.add_argument('--input_ch', default=3, type=int, help='# channels of the input images')
	parser.add_argument('--output_ch', default=3, type=int, help='# channels of the output images')	

	parser.add_argument('--img_loss', type=str, default='l1_loss', help='adversarial loss function, options: mse_loss, l1_loss')
	parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in first conv layer')
	parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in first conv layer')
	parser.add_argument('--num_down', default=4, type=int, help='# downsampling layers in the generator')
	parser.add_argument('--n_layers', default=3, type=int, help='# layers in the discriminator')

	opt = parser.parse_args()

	opt.img_loss_f = utils.get_function(opt.img_loss)
	opt.ssim_f = pytorch_ssim.ssim


	opt.chkpnt_dir = os.path.join(opt.exp_dir,'checkpoints')

	opt.im_path = os.path.join(opt.exp_dir, 'test_images')
	os.system(f"mkdir {opt.im_path}")


	opt.log_file = f'{opt.exp_dir}/log_file_test.txt'


	cfg = ''
	cfg += '----------------- Options ---------------\n'
	for k, v in sorted(vars(opt).items()):
		cfg += f'{str(k):>25}: {str(v):<30}\n'
	cfg += '----------------- End -------------------'
	print(cfg)
	
	with open(f'{opt.exp_dir}/config_test.txt', 'w') as f:
		f.write(f'{cfg}')

	return opt