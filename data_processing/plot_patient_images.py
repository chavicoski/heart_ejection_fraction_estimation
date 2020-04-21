import sys
import os
from pydicom import dcmread
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <CASE_FOLDER_PATH> [<N_PROCESSES>]")
    sys.exit()

case_path = sys.argv[1] 
case_id = case_path.split("/")[-1]
if len(sys.argv) > 2:
    n_proc = int(sys.argv[2])
else:
    n_proc = mp.cpu_count()

def save_dicom_plot(dicom_f):
    dicom_data = dcmread(dicom_f) 

    if 'PixelData' in dicom_data:
        pixel_array = dicom_data.pixel_array
        plt.imshow(pixel_array, cmap=plt.cm.bone)
        sax_id = dicom_f.split("/")[-2]
        parent_folder = f"plots/images/case_{case_id}/{sax_id}"
        os.makedirs(parent_folder, exist_ok=True)
        img_plot_name = dicom_f.split("/")[-1][:-4]
        plt.savefig(f"{parent_folder}/{img_plot_name}.png")

    else:
        print(f"No image data in dicom file {dicom_f}")

'''
Store all the slices images in the case. Go through al the sax folders of each time step to save the images
'''
for sax_dir in tqdm(glob.glob(os.path.join(case_path, "study/sax_*"))):  # Go through each sax folder in the case
    dicom_files = glob.glob(os.path.join(sax_dir, "*.dcm"))  # Get the list of dicom files in the sax folder
    n_dicom_files = len(dicom_files) 

    if n_dicom_files == 30:
        # Create the plots in parallel
        pool = mp.Pool(processes=n_proc)
        pool.map(save_dicom_plot, dicom_files)

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
