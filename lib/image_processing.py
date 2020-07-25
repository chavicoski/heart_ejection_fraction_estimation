import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, disk
import cv2
import multiprocessing 
from array2gif import write_gif
from pydicom import dcmread

############################
# Detect movement with FFT #
############################

def fourier_time_transform_slice(image_3d):
    '''Returns (width, height) matrix
    3D array -> 2D array (compresses time dimension)
    [slice, height, width] -> [height, width]
    Apply Fourier transform to 3d data (time,height,width) to get 2d data (height, width)
    '''
    # Apply FFT to the selected slice
    fft_image_2d = np.fft.fftn(image_3d)[1, :, :]
    return np.abs(np.fft.ifftn(fft_image_2d))


def fourier_time_transform(patient_slices):
    '''
    list of 3D numpy arrays -> list of 2D numpy arrays (compresses time dimension)
    Apply Fourier transform for analyzing movement over time.
    '''
    # Apply FFT to each slice to see movement over time
    ftt_images = [fourier_time_transform_slice(patient_slice) for patient_slice in patient_slices]
    return ftt_images


############################## 
# Segmentation to detect ROI #
##############################

def threshold_segmentation(patient_image):
    """Returns the segmented binary slice image
    Segmententation of a slice image with otsu filter
    ref: https://en.wikipedia.org/wiki/Otsu’s_Method
    """
    threshold = threshold_otsu(patient_image)
    binary_slice = patient_image > threshold
    return binary_slice

def segment_multiple(patient_images):
    """Returns a list with the segmented binary slices
    Apply the function thresh_segmentation() to every slice image of the patient
    """
    segmented_images = []
    for aux_image in patient_images:
        seg_image = threshold_segmentation(aux_image)
        if seg_image.sum() > seg_image.size * 0.5:
            seg_image = 1 - seg_image
        segmented_images.append(seg_image)

    return segmented_images

def roi_mean_yx(patient_images):
    """Returns mean(y) and mean(x) [double]
    Mean coordinates in segmented patients slices. To identify the ROI
    This function performs erosion to get a better result.
    Original: See https://nbviewer.jupyter.org/github/kmader/Quantitative-Big-Imaging-2019/blob/master/Lectures/06-ShapeAnalysis.ipynb
    """
    seg_images = segment_multiple(patient_images)
    num_images = len(seg_images)
    y_all, x_all = [], []
    neighborhood = disk(2)

    for i, seg_image in enumerate(seg_images):
        # Perform erosion to get rid of wrongly segmented small parts
        seg_images_eroded = binary_erosion(seg_image, neighborhood) 

        # Filter out background of slice, after erosion [background=0, foreground=1]
        y_coord, x_coord = seg_images_eroded.nonzero()

        if len(y_coord) > 0 and len(x_coord) > 0:
            # Save mean coordinates of foreground 
            y_all.append(np.mean(y_coord))
            x_all.append(np.mean(x_coord))

    # Return mean of mean foregrounds - this gives an estimate of ROI coords.
    mean_y = int(np.mean(np.array(y_all)))
    mean_x = int(np.mean(np.array(x_all)))
    return mean_y, mean_x


#######################
# IMAGE NORMALIZATION #
#######################

def histogram_normalize(patient_slices, clip_limit=0.03):
    '''
    Apply image normalization with adaptive equalization (check source) to a list of
    numpy arrays with the slices data from a patient
    Source: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    '''
    norm_images = []
    for aux_slice in patient_slices:
        timesteps = aux_slice.shape[0]
        aux_norm_images = []
        for t in range(timesteps):
            aux_norm_images.append(exposure.equalize_adapthist(aux_slice[t].astype(np.uint16), clip_limit=clip_limit))
        norm_images.append(np.array(aux_norm_images))

    return norm_images


###################
# IMAGE RESCALING #
###################

def rescale_patient_slices(patient_slices, pix_spacings):
    '''Given a list of numpys with the slices of a patient and the pixel spacings
    for each slice, this function resizes the images to match 1mm for each pixel'''
    scaled_slices = []  # To store the rescaled numpys
    for s in range(len(patient_slices)):
        aux_slice = patient_slices[s]
        x_dist, y_dist = pix_spacings[s]
        timesteps = aux_slice.shape[0]
        assert timesteps == 30
        aux_slices = []
        for t in range(timesteps):
            aux_slices.append(cv2.resize(src=aux_slice[t], dsize=None, fx=x_dist, fy=y_dist, interpolation=cv2.INTER_CUBIC))
        scaled_slices.append(np.array(aux_slices))

    return scaled_slices

def resize_patient_slices(patient_slices, target_height, target_width):
    '''Given a list of numpys with the slices of a patient and the target shape
    this function resizes the images to the target shape.
    Returns: One numpy array of shape (num_slices, timesteps, target_height, target_width)'''
    num_slices = len(patient_slices)
    timesteps = patient_slices[0].shape[0]
    resized_images = np.zeros((num_slices, timesteps, target_height, target_width))
    for s in range(num_slices):
        for t in range(timesteps):
            resized_images[s,t] = cv2.resize(patient_slices[s][t], dsize=(target_height, target_width), interpolation=cv2.INTER_CUBIC)

    return resized_images

########
# CROP #
########

def crop_roi(image, dim_y, dim_x, coord_y, coord_x):
    """
    Crops an image from the given coords (coord_y, coord_x), such that the resulting image is of
    dimensions (dim_y, dim_x), i.e. height and width.
    Resulting image is filled out from top-left corner, and remaining pixels are left black.
    """
    coord_y, coord_x = int(round(coord_y)), int(round(coord_x))
    height, width = image.shape
    if dim_x > width or dim_y > height: 
        raise ValueError('Crop dimensions larger than image dimension!')

    crop_image = np.zeros((dim_y, dim_x))
    dx, dy = int(dim_x / 2), int(dim_y / 2)
    dx_odd, dy_odd = int(dim_x % 2 == 1), int(dim_y % 2 == 1)

    # Find boundaries for cropping
    dx_left = max(0, coord_x - dx)
    dx_right = min(width, coord_x + dx + dx_odd)
    dy_up = max(0, coord_y - dy)
    dy_down = min(height, coord_y + dy + dy_odd)

    # Find how many pixels to fill out in new image
    range_x = dx_right - dx_left
    range_y = dy_down - dy_up

    # Fill out new image from top left corner
    # Leave pixels outside range as 0's (black)
    crop_image[0:range_y, 0:range_x] = image[dy_up:dy_down, dx_left:dx_right]
    return crop_image


def crop_heart(patient_slices, target_height=150, target_width=150):
    '''
    Finds the heart and crops it. With the crop size provided (target_height, target_width).
    '''
    # Find center for cropping
    fft_images = fourier_time_transform(patient_slices)
    roi_y, roi_x = roi_mean_yx(fft_images)

    # Create new 4d image array to store the crop
    num_slices = len(patient_slices)
    timesteps = patient_slices[0].shape[0]
    heart_crop_slices = np.zeros((num_slices, timesteps, target_height, target_width))

    # Crop every slice image
    for s in range(num_slices):
        for t in range(timesteps):
            heart_crop_slices[s, t] = crop_roi(patient_slices[s][t], target_height, target_width, roi_y, roi_x)

    return heart_crop_slices


########################
# Preprocess_pipelines #
########################

def preprocess_pipeline0(patient_slices, target_size=(150, 150)):
    '''Basic preprocessing to resize the images to the target_size'''
    resized_images = resize_patient_slices(patient_slices, target_size[0], target_size[1])
    resized_images = resized_images / resized_images.max()  # Normalize [0...1]
    return resized_images

def preprocess_pipeline1(patient_slices, pix_spacings, target_size=(150, 150)):
    '''
    Given a list of numpys with all the slices data from a patient. Returns the preprocessed version
    following this steps:
        1. Rescale the images (1 pixel = 1mm)
        2. Histogram normalization (to balance brightness)
        3. Find ROI and crop it (with fft through time)
    '''
    # 1.
    rescaled_images = rescale_patient_slices(patient_slices, pix_spacings)
    # 2.
    norm_images = histogram_normalize(rescaled_images)
    # 3.
    crop_images = crop_heart(norm_images, target_size[0], target_size[1])

    return crop_images


################
# IO FUNCTIONS #
################

def save_slice(slice_image, path):
    '''
    Auxiliary function that stores the given image (numpy) in the
    given path (including the name of the output file)
    '''
    plt.imshow(slice_image, cmap=plt.cm.bone)
    plt.savefig(path)
            

def gif_preprocessing(slice_images):
    '''
    This function preprocesses the original images of a slice with 1 color 
    channel in order to have 3 channels and the values in the range [0,255]
    to be able to use the library that creates the gif animation
    '''
    timesteps, h, w = slice_images.shape
    norm_slice = (slice_images*255.0) / slice_images.max()  # To range [0,255]
    processed_slice = []  # List with an image for each timestep
    for t in range(timesteps):
        selected_image = norm_slice[t,:,:]
        processed_slice.append([selected_image, selected_image, selected_image])  # 3 channels
    return np.array(processed_slice).astype(np.uint8)


def save_patient_slices(patient_slices, folder_path, n_proc=0):
    '''
    Given a numpy array of shape (n_slices, timesteps, height, with) with the
    data of a patient, this function creates and stores the plots for all the
    slices and timesteps including a gift animation per slice. This images are
    stored in 'folder_path' and if it doesn't exist it will be created. 
    '''
    os.makedirs(folder_path, exist_ok=True)  # Check or create the save path
    n_slices = len(patient_slices) 
    timesteps, h, w = patient_slices[0].shape
    if n_proc < 1: 
        n_proc = multiprocessing.cpu_count()  # Get the number of CPU cores
    for s in range(n_slices):
        slice_path = os.path.join(folder_path, f"slice_{s}")
        os.makedirs(slice_path, exist_ok=True)
        slice_images = patient_slices[s]
        slice_images_paths = [os.path.join(slice_path, f"{i:03}.png") for i in range(slice_images.shape[0])]
        pool_arguments = zip(slice_images, slice_images_paths)
        write_gif(gif_preprocessing(slice_images), os.path.join(slice_path, "animation.gif"), fps=30)
        with multiprocessing.Pool(processes=n_proc) as pool:
            pool.starmap(save_slice, pool_arguments)


def get_dicom_files(slice_path):
    '''Given the path to a slice folder with .dcm files, this function returns
    the list (of length 30) of the valid files from the folder'''
    dicom_files = sorted([f_name for f_name in os.listdir(slice_path) if f_name.endswith(".dcm")])
    if len(dicom_files) == 30:
        return dicom_files
    elif len(dicom_files) > 30:
        get_timestep = lambda name: int(name.split("-")[3][:-4])  # Order by acquisition index
        selected_files = sorted(dicom_files, key=get_timestep)[-30:]  # Select the images from the last acquisition
        return sorted(selected_files)
    else:
        return dicom_files + (["<BLACK>"] * (30-len(dicom_files))) # Fill with black images
            

def get_patient_slices(case_number, split, data_path="../cardiac_dataset/"):
    '''
    Given a patient number and the partition that belongs returns a list of numpy arrays
    each of them representing the images for a slice trough time (shape: (timesteps, height, width))
    and a list with the pixel spacing for each slice (used for preprocessing)
    '''
    patient_path = os.path.join(data_path, f"{split}/{split}/{case_number}/study")
    patient_slices = []
    pix_spacings = []
    for sax_folder in sorted(os.listdir(patient_path)):
        if sax_folder.startswith("sax_"):
            aux_shape = None
            pix_spacing = None
            sax_images = []
            sax_path = os.path.join(patient_path, sax_folder)
            dicom_files = get_dicom_files(sax_path)
            if len(dicom_files) == 0: return None, None
            for dicom_name in dicom_files:
                if dicom_name == "<BLACK>":
                    sax_images.append(np.zeros(aux_shape))
                else:
                    dicom_data = dcmread(os.path.join(sax_path, dicom_name))
                    image_array = dicom_data.pixel_array
                    if aux_shape == None:
                        aux_shape = image_array.shape
                    elif aux_shape != image_array.shape:
                        print(f"get_patient_slices(): Error! The slices shapes don't match {aux_shape} != {image_array.shape} (case {case_number})")
                        return None, None
                    sax_images.append(image_array)
                    if pix_spacing == None:
                        pix_spacing = dicom_data.PixelSpacing
                    elif pix_spacing != dicom_data.PixelSpacing:
                        print(f"get_patient_slices(): Warning! The pixel spacings don't match {pix_spacing} != {dicom_data.PixelSpacing} (case {case_number})")

            patient_slices.append(np.array(sax_images))
            pix_spacings.append(pix_spacing)

    return patient_slices, pix_spacings

##################
# TEST FUNCTIONS #
##################

if __name__ == "__main__":
    import sys
    from time import time

    print("### PREPROCESS FUNCTIONS TESTS ###")

    ######################################
    # TEST OF THE PREPROCESSING PIPELINE #
    ######################################
    split = "train"   # Split of the case to process
    case_number = 39
    target_size = (150, 150)
    patient_slices, pix_spacings = get_patient_slices(case_number, split)
    start = time()
    save_patient_slices(patient_slices, f"plots/images/case_{case_number}")  # Store the images of the case
    end = time()
    print(f"Time elapsed during orig plot: {end-start:.2f} seconds")
    start = time()
    #preproc_patient = preprocess_pipeline0(patient_slices, target_size=target_size)  # Do preprocesing
    preproc_patient = preprocess_pipeline1(patient_slices, pix_spacings, target_size=target_size) # Do preprocesing
    end = time()
    print(f"Time elapsed during processing: {end-start:.2f} seconds")
    start = time()
    save_patient_slices(preproc_patient, f"plots/images/case_{case_number}_preproc")  # Store preprocessed images of the case
    end = time()
    print(f"Time elapsed during preproc plot: {end-start:.2f} seconds")
