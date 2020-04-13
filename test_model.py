import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import argparse
import numpy as np 
from tqdm import tqdm
import utils
import options
import os
import get_data
import networks
import time


opt = options.test_options()
np.random.seed(1234)


files = os.listdir(opt.image_dir)
test_patients = [f for f in files if f[0] == 'A' and not '.zip' in f]


# get the datasets ready for iterations
test_dataset = get_data.TestDatasetFromFolder(opt.image_dir, test_patients, opt.num_cross, opt.num_source, opt.input_ch, opt.output_ch, \
										   opt.log_file, opt.height_image, opt.cross_ID, opt.test_target, transform=transforms.Compose([
																																transforms.ToTensor()
																																# , normalize
																																]))


test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.workers)

print('length of test dataset: ', len(test_dataset))


netG = networks.Generator(opt).cuda()
# print('\nGenerator:\n', netG)
netD = networks.NLayerDiscriminator(opt).cuda()
# print('\nDiscriminator:\n', netD)
Conv3D = networks.Conv3DBlock(opt).cuda()
# print('\nConv3D blocks:\n', Conv3D)


print('\n\n\nUploading model...')
model_path = os.path.join(opt.chkpnt_dir, 'model_best.pth.tar')

if os.path.isfile(model_path):
	utils.print_and_save_msg(f"=> loading Model '{model_path}'", opt.log_file)
	checkpoint = torch.load(model_path) 
	netG.load_state_dict(checkpoint['Generator'])
	# netD.load_state_dict(checkpoint['Discriminator'])
	Conv3D.load_state_dict(checkpoint['Conv3D'])
	utils.print_and_save_msg(f"=> loaded checkpoint '{model_path}' for testing...\n\n", opt.log_file)
else:
	print(f"=> no checkpoint found at '{model_path}', exiting from the program...")



t = tqdm(iter(test_loader), leave=False, total=len(test_loader))

im_loss, ssim, counter, total_images_processed = 0, 0, 0, 0

st_time = time.time()
for i, (source, target, target_path) in enumerate(t):

	# change range to [-1.0, 1.0]
	source = (source-0.5)/0.5
	target = (target-0.5)/0.5

	source = source.cuda()
	target = target.cuda()

	src = Conv3D(source)
	image_fake = netG(src)

	SSIM = opt.ssim_f(image_fake*0.5+0.5, target*0.5+0.5)
	Image_loss = opt.img_loss_f(image_fake, target)

	im_loss +=  Image_loss.item()
	ssim += SSIM.item()

	counter += 1

	total_images_processed += len(target)

	for ii in range(image_fake.shape[0]):

		sample_target_1 = utils.convert_to_numpy(target[ii])
		sample_fake_1 = utils.convert_to_numpy(image_fake[ii])

		f_name, file_extension = os.path.splitext(target_path[ii])
		filename = f"{f_name.split('/')[-1]}__{f_name.split('/')[-2]}"
		# print(filename)
		utils.save_images(sample_target_1, sample_fake_1, name=filename, ext=file_extension, im_path=opt.im_path,  mode='test')

	# break

im_loss /= counter
ssim /= counter

msg = f'\n\n\n\ntotal pairs processed: {total_images_processed:} | ' \
	+ f'Image loss : {im_loss:.4f} | SSIM : {ssim:.4f} | ' \
	+ f'\tin {int(time.time() - st_time):05d} in seconds\n\n\n'

utils.print_and_save_msg(msg, opt.log_file)