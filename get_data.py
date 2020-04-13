import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import numpy as np 
import torch
import matplotlib.pyplot as plt 
import utils 


HEIGHT = 400 # height of the images (crop the rest which is black, scale bar etc.)

class DatasetFromFolder(Dataset):
	def __init__(self, image_dir, patient_list, num_cross, num_source, input_ch, output_ch, log_file, height_image, transform=None):
		super(DatasetFromFolder, self).__init__()		

		self.data_pairs = [] # list of images, num_source+1 images, source images + target image
		self.num_source = num_source
		self.image_dir = image_dir
		self.transform = transform
		self.input_ch = input_ch
		self.output_ch = output_ch
		self.log_file = log_file
		self.height_image = height_image

		measurements = []
		meas_numbers = []
		D_patient = {}

		Images = []
		img_numbers = []

		for p in patient_list:
			meas_ = os.listdir(f'{image_dir}/{p}')
			meas = [x for x in meas_ if x[0] == '2']
			meas_numbers.append(len(meas))
			if len(meas) < num_source+1: # we need num_source images as input and 1 additional image as output/target.
				utils.print_and_save_msg(f'The patient {p} is discarded from the dataset. no enough measurements', self.log_file)
				continue
			
			D_patient[p] = meas
			for m in meas:
				# print(f'{image_dir}/{p}/{m}')
				measurements.append(f'{image_dir}/{p}/{m}')

		for ms in measurements:
			list_ = os.listdir(ms)
			img_list = [f'{ms}/{x}' for x in list_]
			Images += img_list
			img_numbers.append(len(img_list))  

		# print('Images length: ', len(Images))
		# print(Images)
		# exit()
		# png_list = [x for x in Images if '.png' in x]
		# jpg_list = [x for x in Images if '.jpg' in x]
		# print(f'PNG: {len(png_list)}, FPG: {len(jpg_list)}, TOTAL: {len(png_list)+len(jpg_list)} images')

		utils.print_and_save_msg(f'Minimum number of cross-sections in a measurement folder: {min(img_numbers)}\nMaximum number of cross-sections in a measurement folder: {max(img_numbers)}', self.log_file)
		utils.print_and_save_msg(f'Set of measurements: {set(img_numbers)}', self.log_file)

		for p in D_patient:
			for c in range(num_cross):        
				meas_dates = D_patient[p]					
				## patient's images for this cross section:
				images = [x for x in Images if p in x and (f'{c:03d}.jpg' in x or f'{c:03d}.png' in x)]				
				for ii in range(len(images)-num_source):
					im_lst = [images[ii], images[ii+1], images[ii+2], images[ii+3]]
					# if images_ok(im_lst) :
					self.data_pairs.append(im_lst)

	def __getitem__(self, index):

		L = []
		for i in range(self.num_source+1):
			# img_path = os.path.join(self.image_dir, self.data_pairs[index][i])	
			img_path = self.data_pairs[index][i]
			try:			
				im = plt.imread(img_path)
				h, w, ch = im.shape
				crop = im[:self.height_image, h:, :] # cropping the left image (retina)
				L.append(crop)
			except:
				utils.print_and_save_msg(f'\nimage is corrupted: {img_path}\n', self.log_file)
				if len(L) > 0:
					L.append(L[-1])
				
		if len(L) == self.num_source: # that means first image is missing
			L = L[0] + L

		source = np.concatenate(L[:-1], axis=2) # concatenate source images from the channel dimension
		target = L[-1] # get the last image in the list as target image

		if self.transform:
			source = self.transform(source)
			target = self.transform(target)

		return source, target

	def __len__(self):
		return len(self.data_pairs)


class TestDatasetFromFolder(Dataset):
	def __init__(self, image_dir, patient_list, num_cross, num_source, input_ch, output_ch, log_file, height_image, cross_ID, test_target, transform=None):
		super(TestDatasetFromFolder, self).__init__()		

		self.data_pairs = [] # list of images, num_source+1 images, source images + target image
		self.num_source = num_source
		self.image_dir = image_dir
		self.transform = transform
		self.input_ch = input_ch
		self.output_ch = output_ch
		self.log_file = log_file
		self.height_image = height_image

		measurements = []
		meas_numbers = []
		D_patient = {}

		Images = []
		img_numbers = []

		for p in patient_list:
			meas_ = os.listdir(f'{image_dir}/{p}')
			meas = [x for x in meas_ if x[0] == '2']
			meas_numbers.append(len(meas))
			if len(meas) < num_source+1: # we need num_source images as input and 1 additional image as output/target.
				utils.print_and_save_msg(f'The patient {p} is discarded from the dataset. no enough measurements', self.log_file)
				continue
			
			D_patient[p] = meas
			for m in meas:
				# print(f'{image_dir}/{p}/{m}')
				measurements.append(f'{image_dir}/{p}/{m}')

		for ms in measurements:
			list_ = os.listdir(ms)
			img_list = [f'{ms}/{x}' for x in list_]
			Images += img_list
			img_numbers.append(len(img_list))  

		# print('Images length: ', len(Images))
		# print(Images)
		# exit()
		# png_list = [x for x in Images if '.png' in x]
		# jpg_list = [x for x in Images if '.jpg' in x]
		# print(f'PNG: {len(png_list)}, FPG: {len(jpg_list)}, TOTAL: {len(png_list)+len(jpg_list)} images')

		utils.print_and_save_msg(f'Minimum number of cross-sections in a measurement folder: {min(img_numbers)}\nMaximum number of cross-sections in a measurement folder: {max(img_numbers)}', self.log_file)
		utils.print_and_save_msg(f'Set of measurements: {set(img_numbers)}', self.log_file)

		### all cross-sections testing
		if test_target == 'all':
			# print('\n\nhere\n\n\n')
			for p in D_patient:
				for c in range(num_cross):        
					meas_dates = D_patient[p]					
					## patient's images for this cross section:
					images = [x for x in Images if p in x and (f'{c:03d}.jpg' in x or f'{c:03d}.png' in x)]				
					for ii in range(len(images)-num_source):
						im_lst = [images[ii], images[ii+1], images[ii+2], images[ii+3]]
						# if images_ok(im_lst) :
						self.data_pairs.append(im_lst)

		### single cross-section testing
		elif test_target == 'single':
			for p in D_patient:
				meas_dates = D_patient[p]					
				## patient's images for this cross section:
				# for im_ in Images:
				# 	print(im_)
				# 	print('\n\n')
				# exit()
				# images = [x for x in Images if p in x and (f'{cross_ID:03d}.jpg' in x or f'{cross_ID:03d}.png' in x)]				
				images = [x for x in Images if p in x and ('.jpg' in x or '.png' in x)]				
				for ii in range(len(images)-num_source):
					im_lst = [images[ii], images[ii+1], images[ii+2], images[ii+3]]
					# if images_ok(im_lst) :
					self.data_pairs.append(im_lst)


	def __getitem__(self, index):

		L = []
		for i in range(self.num_source+1):
			# img_path = os.path.join(self.image_dir, self.data_pairs[index][i])	
			img_path = self.data_pairs[index][i]
			try:			
				im = plt.imread(img_path)
				h, w, ch = im.shape
				crop = im[:self.height_image, h:, :] # cropping the left image (retina)
				L.append(crop)
			except:
				utils.print_and_save_msg(f'\nimage is corrupted: {img_path}\n', self.log_file)
				if len(L) > 0:
					L.append(L[-1])
				
		if len(L) == self.num_source: # that means first image is missing
			L = L[0] + L

		source = np.concatenate(L[:-1], axis=2) # concatenate source images from the channel dimension
		target = L[-1] # get the last image in the list as target image

		if self.transform:
			source = self.transform(source)
			target = self.transform(target)

		return source, target, self.data_pairs[index][-1] # return path of the target image also

	def __len__(self):
		return len(self.data_pairs)
