# CS168

Hi,
this is the first version of the code. 

Prerquisite packages:
In order to run the program (main.py) in Python, there are some packages (PyTorch, NumPy etc.) are required and they are in the environment.yml file. ([setting up environment using Anacond](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))


Assumption (Hypothesis):
We claim that a future cross section of the retina can be interpolated from past measurements of the same cross sections. Each measurmenent, except for the last one
Here, I set the number of previous measurements to 3 ("opt.num_source" argument in the options.py), we can play it.



Dataset:
Data files should be under the "image_dir" argument that is set in "options.py". The image files of patients AGPS023, AGPS024, AGPS025, AGPS027, and AGPS028 are corrupted (0 KB images).
You can change the train/val fractions in the fiest lines of the main.py. I initially set to 0.8/0.2. Program automatically lists the patients and sparetes the 80% of the patients for training etc.
And, patients who have less than "num_source" measurements are discarded, not included to datasets. "num_source" is the number of the past measurments used to predict to future measurements. Dataset is thought as (source, target) pairs, where source is the concatenated num_source images, while target is a single image. (Each image has 3 channels, RGB).



GAN Architecture:
Generator applies 2D Conv on the source image in a U-Net type, which has 2 maing sub-networks: downsampling and upsampling. And, skip connections betweek these two part copies features from the front part of the network to the rear part. 
Discriminator is copied from a well-known project called pix2pix. We can change its number of layers in accordance w/ the training results.
We can think different architectures in future.


Code creates a project folder w/ the name of the current date, hour etc. Each project folder has its own configuration file (storing the implementation details), images folder (showing image samples), log_file/losses_over_epochs files (showing/displaying the progress of the losses throughout the training).




