import sys
import os
import pandas as pd
import numpy as np
from pydicom import dcmread
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob

# Root data path
dataset_path = "../cardiac_dataset"
# Partitons paths
train_path = os.path.join(dataset_path, "train")
dev_path = os.path.join(dataset_path, "validate")
test_path = os.path.join(dataset_path, "test")
# Partitons data paths
train_data_path = os.path.join(train_path, "train")
dev_data_path = os.path.join(dev_path, "validate")
test_data_path = os.path.join(test_path, "test")
# Partitons CSV paths
train_csv_path = os.path.join(train_path, "train.csv")
dev_csv_path = os.path.join(dev_path, "validate.csv")
test_csv_path = os.path.join(test_path, "solution.csv")
# Load CSV's
train_df = pd.read_csv(train_csv_path)
dev_df = pd.read_csv(dev_csv_path)
test_df = pd.read_csv(test_csv_path)

#################################################
# GO THROUGH ALL THE DATASET TO DO THE ANALYSIS #
#################################################
'''
Info:
- Each patient has N sax folders (1 per slice) where each folder should have 30 dicom images for each timestep.

- The folders with more or less than 30 slices will be ignored.
'''
ignored_slices = []
pixels_stats   = {"train": [], "dev": [], "test": []}  # Tuples of pixels values (avg, max, min)
slices_count   = {"train": [], "dev": [], "test": []}  # Number of slices per patient
spacing_count  = {"train": [], "dev": [], "test": []}  # Tuples with spaces in each dimension (h, w, depth)
shapes_count   = {"train": {}, "dev": {}, "test": {}}  # To store the different shapes in the data -> key: shape tuple string, value: number of adquisitions
systole_count  = {"train": [], "dev": [], "test": []}  # List with float values for systole labels
diastole_count = {"train": [], "dev": [], "test": []}  # List with float values for diastole labels
data_splits    = [("train", train_data_path), ("dev", dev_data_path), ("test", test_data_path)]

for split_name, data_path in data_splits:
    for patient in tqdm(os.listdir(data_path), desc=f"{split_name + ' split':<13}"):
        sax_dirs = glob.glob(os.path.join(data_path, patient, "study/sax_*"))  # Get all the slices folders of type sax
        n_valid_slices = 0  # To count the number of slices per patient
        first = True  # Take some data just from the first slice
        for sax_dir in sax_dirs:
            dicom_files = glob.glob(os.path.join(sax_dir, "*.dcm"))  # Get the list of dicom files in the sax folder
            n_dicom_files = len(dicom_files) 

            if n_dicom_files == 30:
                img_shape = (-1, -1)
                for dicom_f in sorted(dicom_files):
                    dicom_data = dcmread(dicom_f) 
                    pixel_array = dicom_data.pixel_array
                    if first:
                        first = False
                        # Get the spacing between pixels to scale the images
                        x_dist, y_dist = dicom_data.PixelSpacing  # Row and column spacing
                        x_dist, y_dist = float(x_dist), float(y_dist)
                        z_dist = float(dicom_data.SliceThickness)  # Depth spacing
                        spacing_count[split_name].append((x_dist, y_dist, z_dist))
                        # Store slices shape
                        img_shape = pixel_array.shape
                        shapes_count[split_name][str(img_shape)] = shapes_count[split_name].get(str(img_shape), 0) + 1  

                    # Get pixel values stats
                    pix_avg = np.mean(pixel_array)
                    pix_max = np.max(pixel_array)
                    pix_min = np.min(pixel_array)
                    pixels_stats[split_name].append((pix_avg, pix_max, pix_min))

                n_valid_slices += 1

            else:
                ignored_slices.append((sax_dir, n_dicom_files))

        if n_valid_slices > 0:
            slices_count[split_name].append(n_valid_slices)
        else:
            print(f"There are no valid slices for patient {patient} of {split_name} split!")

# Get stats from labels
for split_name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
    if split_name != "test":
        for idx, row in df.iterrows():
            systole_count[split_name].append(row["Systole"]) 
            diastole_count[split_name].append(row["Diastole"]) 
    else:  # Test dataframe has diferent format
        for idx, row in df.iterrows():
            if row["Id"].endswith("Systole"):
                systole_count[split_name].append(row["Volume"]) 
            elif row["Id"].endswith("Diastole"):
                diastole_count[split_name].append(row["Volume"]) 


##############
# SHOW STATS #
##############

'''
Slices stats
'''
print("\nSlices count by patient:")
for split, counts in slices_count.items():
    print(f"{split}: average={sum(counts)/len(counts):.2f}, max={max(counts)}, min={min(counts)}")

# Histogram version of slices counts
plt.hist([x[1] for x in slices_count.items()], bins=30, label=[x[0] for x in slices_count.items()])
plt.legend(loc="upper right")
plt.xlabel("N slices")
plt.ylabel("Count")
plt.title(f"Count of slices by patient for each partition")
plt.savefig(f"plots/slices_count.png")
plt.clf()  # Reset figure for next plot

print(f"\nIngnored slices ({len(ignored_slices)}):")
for sax_dir, n_dicom_files in ignored_slices:
    print(f"{sax_dir} -> {n_dicom_files} dicom files")

'''
Shapes stats
'''
print("\nShapes:")
for split, shapes in shapes_count.items():
    print(f"{split} split:")
    for shape, count in sorted(shapes.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{shape}: {count:>4}")

'''
Spacing stats
'''
print("\nPixels spacings:")
for split, counts in spacing_count.items():
    x_avg, y_avg, z_avg = 0, 0, 0
    x_max, y_max, z_max = -1, -1, -1
    x_min, y_min, z_min = 999, 999, 999
    total = 0
    for x_dist, y_dist, z_dist in counts:
        if x_dist > x_max: x_max = x_dist
        if y_dist > y_max: y_max = y_dist
        if z_dist > z_max: z_max = z_dist

        if x_dist < x_min: x_min = x_dist
        if y_dist < y_min: y_min = y_dist
        if z_dist < z_min: z_min = z_dist

        x_avg += x_dist
        y_avg += y_dist
        z_avg += z_dist
        total += 1

    print(f"\t{split} split:")
    print(f"\t\tx_dist -> avg={x_avg/total} - max={x_max} - min={x_min}")
    print(f"\t\ty_dist -> avg={y_avg/total} - max={y_max} - min={y_min}")
    print(f"\t\tz_dist -> avg={z_avg/total} - max={z_max} - min={z_min}")

x_dists = [dist[0] for split, counts in spacing_count.items() for dist in counts]
y_dists = [dist[1] for split, counts in spacing_count.items() for dist in counts]
z_dists = [dist[2] for split, counts in spacing_count.items() for dist in counts]

# Row spacings histogram
plt.hist(x_dists, bins=30)
plt.xlabel("Distance")
plt.ylabel("Count")
plt.title(f"Count of row spacings")
plt.savefig(f"plots/spacings_x.png")
plt.clf()  # Reset figure for next plot

# Column spacings histogram
plt.hist(y_dists, bins=30)
plt.xlabel("Distance")
plt.ylabel("Count")
plt.title(f"Count of column spacings")
plt.savefig(f"plots/spacings_y.png")
plt.clf()  # Reset figure for next plot

# Depth spacings histogram
plt.hist(z_dists, bins=30)
plt.xlabel("Distance")
plt.ylabel("Count")
plt.title(f"Count of depth spacings")
plt.savefig(f"plots/spacings_z.png")
plt.clf()  # Reset figure for next plot

'''
Pixels stats
'''
print("\nPixels values:")
for split, pix_stats in pixels_stats.items():
    aux_avg = 0
    aux_max = -1
    aux_min = 999999
    total = 0
    for pix_avg, pix_max, pix_min in pix_stats:
        if pix_max > aux_max: aux_max = pix_max 
        if pix_min < aux_min: aux_min = pix_min 

        aux_avg += pix_avg
        total += 1

    print(f"\t{split} split:")
    print(f"\t\taverage_pixel={aux_avg/total} - max_pixel={aux_max} - min_pixel={aux_min}")

pix_avgs = [stat[0] for split, pix_stats in pixels_stats.items() for stat in pix_stats]
pix_maxs = [stat[1] for split, pix_stats in pixels_stats.items() for stat in pix_stats]
pix_mins = [stat[2] for split, pix_stats in pixels_stats.items() for stat in pix_stats]

# Pixels averages (by slice and timestep) histogram
plt.hist(pix_avgs, bins=30)
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.title(f"Count of average pixel value for every dicom image")
plt.savefig(f"plots/pixels_averages.png")
plt.clf()  # Reset figure for next plot

# Pixels maximums histogram
plt.hist(pix_maxs, bins=30)
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.title(f"Count of maximum pixel values for every dicom image")
plt.savefig(f"plots/pixels_maxspng")
plt.clf()  # Reset figure for next plot

# Pixels minimums histogram
plt.hist(pix_mins, bins=30)
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.title(f"Count of minimum pixel values for every dicom image")
plt.savefig(f"plots/pixels_mins.png")
plt.clf()  # Reset figure for next plot

'''
Labels stats
'''
total_systole_values = []
total_diastole_values = []
print("\nLabels values:")
for split in systole_count.keys():
    print(f"{split} split:")
    systole_values = systole_count[split]
    diastole_values = diastole_count[split]
    total_systole_values += systole_values
    total_diastole_values += diastole_values
    print(f"\tSystole : average={sum(systole_values)/len(systole_values):.1f} - max={max(systole_values):.1f} - min={min(systole_values):.1f}")
    print(f"\tDiastole: average={sum(diastole_values)/len(diastole_values):.1f} - max={max(diastole_values):.1f} - min={min(diastole_values):.1f}")

# Histogram versions of labels stats
plt.hist([total_systole_values, total_diastole_values], bins=60, label=["Systole", "Diastole"])
plt.legend(loc="upper right")
plt.xlabel("Volume")
plt.ylabel("Count")
plt.title(f"Count of systole and diastole values for all the dataset")
plt.savefig(f"plots/labels_count.png")
plt.clf()  # Reset figure for next plot
