import torch
from torchvision.utils import save_image
import time 
import numpy as np 
import utils
from tqdm import tqdm

def propagate(loader, epoch, netG, netD, Conv3D, optimizer_G, optimizer_D, opt, mode='train'):

	if mode == 'train':
		Conv3D.train()
		netG.train()
		netD.train()
	elif mode == 'valid':
		Conv3D.eval()
		netG.eval()
		netD.eval()

	gen_loss, dis_loss, im_loss, ssim_loss, counter, total_images_processed = 0, 0, 0, 0, 0, 0
	st_time = time.time()
	label_real = torch.tensor(1.0).cuda()
	label_fake = torch.tensor(0.0).cuda()

	t = tqdm(iter(loader), leave=False, total=len(loader))
	for i, (source, target) in enumerate(t):

		# change range to [-1.0, 1.0]
		source = (source-0.5)/0.5
		target = (target-0.5)/0.5

		source = source.cuda()
		target = target.cuda()

		# print('\n')
		# print(f'source shape: {source.shape}\ntarget shape: {target.shape}')
		# print('source min-max: ', source.min(), source.max())
		# print('target min-max: ', target.min(), target.max())
		# print('\n')
		# exit()
		src = Conv3D(source)
		# print(f'\n\n\nAfter 3D Conv, shape: {src.shape}')
		image_fake = netG(src)

		# print(f'\n\nGenerator input (source) shape: {source.shape}\n output (image fake) shape', image_fake.shape)		
		# exit()
		pred_fake = netD(torch.cat((source, image_fake.detach()), 1)) # concatenated images as the input of patch-GAN discriminator
		pred_real = netD(torch.cat((source, target), 1))

		D_fake_loss = opt.adv_loss_f(pred_fake, label_fake.expand_as(pred_fake)) 
		D_real_loss = opt.adv_loss_f(pred_real, label_real.expand_as(pred_real)) 

		loss_D = 0.5 * (D_fake_loss + D_real_loss)

		Image_loss = opt.img_loss_f(image_fake, target)
		SSIM_loss = 1.0 - opt.ssim_f(image_fake*0.5+0.5, target*0.5+0.5)

		gen_pred_fake = netD(torch.cat((source, image_fake), 1))

		G_disc_loss = opt.adv_loss_f(gen_pred_fake, label_real.expand_as(gen_pred_fake)) 

		loss_G = opt.im_coeff * Image_loss + opt.ssim_coeff * SSIM_loss + G_disc_loss

		### backpropagate
		if mode == 'train':

			# utils.set_requires_grad(netD, True) 
			optimizer_D.zero_grad()
			loss_D.backward()			
			# clip_grad_norm_(netD.parameters(), 0.5)
			optimizer_D.step()

			# utils.set_requires_grad(netD, False)
			optimizer_G.zero_grad()
			loss_G.backward()
			# clip_grad_norm_(netG.parameters(), 0.5)
			optimizer_G.step()

		gen_loss += G_disc_loss.item()
		dis_loss += loss_D.item()
		im_loss +=  Image_loss.item()
		ssim_loss += SSIM_loss.item()

		counter += 1

		total_images_processed += len(target)

		# break


	"""
	----------------------------------------------------------------------------------------------------------
	Print messages to the screen and save the progress as png files. Also, save example images as source-target pairs
	 
	"""

	gen_loss /= counter
	dis_loss /= counter
	im_loss /= counter
	ssim_loss /= counter

	if mode == 'train':
		msg = '\n\n'
	else:
		msg = '\n'

	msg += f'{mode}: {epoch:04}/{opt.max_epoch:04}\ttotal pairs processed: {total_images_processed:} | ' \
		+ f'Image loss : {im_loss:.4f} | SSIM loss : {ssim_loss:.4f} |  Gen loss: {gen_loss:.4f} | ' + f'Disc loss: {dis_loss:.4f}' \
		+ f'\tin {int(time.time() - st_time):05d} in seconds\n'

	# print('\n\nHERERE')
	# print(target[0].detach().cpu().numpy().shape)
	utils.print_and_save_msg(msg, opt.log_file)	

	sample_target_1 = utils.convert_to_numpy(target[0])
	# sample_target_2 = utils.convert_to_numpy(target[1])
	sample_fake_1 = utils.convert_to_numpy(image_fake[0])
	# print('\n\ntarget: ', sample_target_1.min(), sample_target_1.max())
	# print('\n\nfake: ', sample_fake_1.min(), sample_fake_1.max())
	# sample_fake_2 = utils.convert_to_numpy(image_fake[1])

	# utils.save_images(sample_target_1, sample_fake_1, sample_target_2, sample_fake_2, epoch=epoch, im_path=opt.im_path,  mode=mode)
	utils.save_images(sample_target_1, sample_fake_1,  epoch=epoch, im_path=opt.im_path,  mode=mode)
	# save_image(target, f'{opt.im_path}/epoch_{epoch}_{mode}.png', nrow=2, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

	return gen_loss, dis_loss, im_loss

