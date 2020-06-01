import sys
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
from array2gif import write_gif
from pydicom import dcmread

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <CASE_FOLDER_PATH> [<N_PROCESSES>]")
    sys.exit()

'''
Given a path to a case folder, this script creates a folder in /plots/images/ with the png's for
all the images of the case and creates the gif animations for each sax slice
'''

case_path = sys.argv[1] 
case_id = case_path.split("/")[-1]
if len(sys.argv) > 2:
    n_proc = int(sys.argv[2])
else:
    n_proc = mp.cpu_count()

def save_dicom_plot(dicom_f, return_data=False):
    '''
    Auxiliary function to save the dicom images
    '''
    dicom_data = dcmread(dicom_f) 
    pixel_array = dicom_data.pixel_array
    plt.imshow(pixel_array, cmap=plt.cm.bone)
    sax_id = dicom_f.split("/")[-2]
    parent_folder = f"plots/images/case_{case_id}/{sax_id}"
    os.makedirs(parent_folder, exist_ok=True)
    img_plot_name = dicom_f.split("/")[-1][:-4]
    plt.savefig(f"{parent_folder}/{img_plot_name}.png")

    if return_data:
        # Return the image with values in range [0, 255]
        norm_image = (pixel_array * 255.0) / pixel_array.max()
        return np.array([norm_image, norm_image, norm_image]).astype(np.uint8)

'''
Store all the slices images in the case. Go through al the sax folders of each time step to save the images
'''
for sax_dir in tqdm(glob.glob(os.path.join(case_path, "study/sax_*"))):  # Go through each sax folder in the case
    dicom_files = glob.glob(os.path.join(sax_dir, "*.dcm"))  # Get the list of dicom files in the sax folder
    n_dicom_files = len(dicom_files) 

    if n_dicom_files == 30:
        # Create the plots in parallel
        pool = mp.Pool(processes=n_proc)
        time_slices = [pool.apply(save_dicom_plot, args=(dicom_f, True)) for dicom_f in sorted(dicom_files)]
        
        # Create slice gif
        write_gif(time_slices, f'plots/images/case_{case_id}/{sax_dir.split("/")[-1]}/animation.gif', fps=15)

    elif n_dicom_files > 30:
        print(f'This case has {n_dicom_files} slices in sax folder {sax_dir.split("/")[-1]}')
        n_adquisitions = int(n_dicom_files / 30)

        # Get the set of dicoms for each adquisition 
        for i in range(1, n_adquisitions + 1):
            # Get the dicoms of the current sax adquisition
            adquisition_files = list(filter(lambda x: x.endswith(f"{i}.dcm"), dicom_files))
            # Create the plots in parallel
            pool = mp.Pool(processes=n_proc)
            pool.map(save_dicom_plot, adquisition_files)

    else:
        print(f'This case has {n_dicom_files} slices in sax folder {sax_dir.split("/")[-1]}')
        # Create the plots in parallel
        pool = mp.Pool(processes=n_proc)
        pool.map(save_dicom_plot, dicom_files)
