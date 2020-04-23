import numpy as np
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, disk


############################
# Detect movement with FFT #
############################

def fourier_time_transform_slice(image_3d):
    '''Returns (width, height) matrix
    3D array -> 2D array
    [slice, height, width] -> [height, width]
    Apply Fourier transform to 3d data (time,height,width) to get 2d data (height, width)
    '''
    # Apply FFT to the selected slice
    fft_image_2d = np.fft.fftn(image_3d)[1, :, :]
    return np.abs(np.fft.ifftn(fft_image_2d))


def fourier_time_transform(patient_slices):
    '''
    4D array -> 3D array (compresses time dimension)
    [slice, time, height, width] -> [slice, height, width]
    Apply Fourier transform for analyzing movement over time.
    '''
    # Apply FFT to each slice to see movement over time
    ftt_image = np.array([fourier_time_transform_slice(patient_slice) for patient_slice in patient_slices])
    return ftt_image


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
    """Returns list with segmented binary slices
    Apply the function thresh_segmentation() to every slice image of the patient
    """
    num_images, height, width = patient_images.shape
    segmented_images = np.zeros((num_images, height, width))

    for i in range(num_images):
        seg_image = thresh_segmentation(patient_images[i])
        if seg_image.sum() > seg_image.size * 0.5:
            seg_image = 1 - seg_image
        segmented_images[i] = seg_image

    return segmented_images

def roi_mean_yx(patient_image):
    """Returns mean(y) and mean(x) [double]
    Mean coordinates in segmented patients slices. To identify the ROI
    This function performs erosion to get a better result.
    Original: See https://nbviewer.jupyter.org/github/kmader/Quantitative-Big-Imaging-2019/blob/master/Lectures/06-ShapeAnalysis.ipynb
    """
    seg_images = segment_multiple(patient_image)
    num_images = seg_images.shape[0]
    y_all, x_all = np.zeros(num_images), np.zeros(num_images)
    neighborhood = disk(2)

    for i, seg_image in enumerate(seg_images):
    # Perform erosion to get rid of wrongly segmented small parts
    seg_images_eroded = binary_erosion(seg_image, neighborhood) 

    # Filter out background of slice, after erosion [background=0, foreground=1]
    y_coord, x_coord = seg_images_eroded.nonzero()

    # Save mean coordinates of foreground 
    y_all[i], x_all[i] = np.mean(y_coord), np.mean(x_coord)

    # Return mean of mean foregrounds - this gives an estimate of ROI coords.
    mean_y = int(np.mean(y_all))
    mean_x = int(np.mean(x_all))
    return mean_y, mean_x


#######################
# IMAGE NORMALIZATION #
#######################

def histogram_normalize_4d(images, clip_limit=0.03):
    '''
    Apply image normalization with adaptive equalization (check source)
    Source: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    '''
    slices, timesteps, _, _ = images.shape
    norm_images_4d = np.empty(images.shape)
    for i in range(slices):
        for j in range(timesteps):
            norm_imgs_4d[i,j] = exposure.equalize_adapthist(images[i,j].astype(np.uint16), clip_limit=clip_limit)

    return norm_images_4d


###################
# IMAGE RESCALING #
###################

def rescale_patient_slices(patient_slices, x_dist, y_dist):
    num_slices, timesteps, _, _ = patient_slices.shape

    # Rescale the first 2d image, in order to find out the resulting dimensions
    aux_image = cv2.resize(src=patient_slices[0, 0], dsize=None, fx=x_dist, fy=y_dist)
    target_height, target_width = aux_image.shape
    scaled_images = np.zeros((num_slices, timesteps, target_height, target_width))

    # Resize for each slice and time step all the images of the patient
    for s in range(num_slices):
        for t in range(timestep):
            scaled_images[s,t] = cv2.resize(src=patient_slices[s,t], dsize=None, fx=x_dist, fy=y_dist)

    return scaled_images

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
    crop_image[0:range_y, 0:range_x] = img[dy_up:dy_down, dx_left:dx_right]
    return crop_img


def crop_heart(patient_slices, target_height=200, target_width=200):
    '''
    Finds the heart and crops it. With the crop size provided (target_height, target_width).
    '''
    # Find center for cropping
    fft_images = fourier_time_transform(patient_slices)
    roi_y, roi_x = roi_mean_yx(fft_images)

    # Create new 4d image array to store the crop
    num_slices, timesteps, _, _ = patient_slices.shape
    heart_crop_slices = np.zeros((num_slices, timesteps, target_height, target_width))

    # Crop every slice image
    for s in range(num_slices):
        for t in range(timesteps):
            heart_crop_slices[s, t] = crop_roi(patient_slices[s, t], target_height, target_width, roi_y, roi_x)

    return heart_crop_slices


########################
# Preprocess_pipelines #
########################

def preprocess_pipeline1(patient_slices, pix_spacings, target_size=(200, 200)):
    '''
    Given a raw patient numpy with all the slices data. Returns the preprocessed version
    following this steps:
        1. Rescale the images (1 pixel = 1mm)
        2. Histogram normalization (to balance brightness)
        3. Find ROI and crop it (with fft through time)
    '''
    # 1.
    rescaled_images = rescale_patient_slices(patient_slices, pix_spacings[0], pix_spacings[1])
    # 2.
    norm_images = histogram_normalize_4d(rescaled_images)
    # 3.
    crop_images = crop_heart(norm_images, target_size[0], target_size[1])

    return crop_images


##################
# TEST FUNCTIONS #
##################

if __name__ == "__main__":
    print("### PREPROCESS FUNCTIONS TESTS ###")
