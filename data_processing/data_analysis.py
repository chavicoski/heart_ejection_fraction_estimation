import sys
sys.path.insert(1, '.')  # To access the libraries
import os
from statistics import mean, median, mode
import pandas as pd
import numpy as np
from pydicom import dcmread
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
from lib.image_processing import get_dicom_files

# Root data path
dataset_path = "../cardiac_dataset"
# Output path
out_path = "plots/analysis"
os.makedirs(out_path, exist_ok=True)
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
- The slices with less than 30 dicoms have <BLACK> synthetic slices to fill the frames up to 30.
- In the case of having more than 30 frames, we take the last 30 of them that belongs to the same adquisition. Because
  the cases with more than 30 have several adquisitions so the number of frames are a multiple of 30.
'''
data_splits    = [("train", train_data_path), ("dev", dev_data_path), ("test", test_data_path)]
systole_count  = {"train": [], "dev": [], "test": []}  # List with float values for systole labels
diastole_count = {"train": [], "dev": [], "test": []}  # List with float values for diastole labels
# SAX 
pixels_stats   = {"train": [], "dev": [], "test": []}  # Tuples of pixels values (avg, max, min)
slices_count   = {"train": [], "dev": [], "test": []}  # Number of slices per patient
frames_count   = {"train": [], "dev": [], "test": []}  # Number of frames per slice 
spacing_count  = {"train": [], "dev": [], "test": []}  # Tuples with spaces in each dimension (h, w, depth)
shapes_count   = {"train": {}, "dev": {}, "test": {}}  # To store the different shapes in the data -> key: shape tuple string, value: number of adquisitions
# CH
two_ch_count   = {"train": [], "dev": [], "test": []}  # Number of 2ch slices per patient
four_ch_count  = {"train": [], "dev": [], "test": []}  # Number of 4ch slices per patient
miss_ch_count  = {"train": [], "dev": [], "test": []}  # Available CH slices for the patients that miss at least one of them
frames_count_2ch = {"train": [], "dev": [], "test": []}  # Number of frames per slice 
frames_count_4ch = {"train": [], "dev": [], "test": []}  # Number of frames per slice 
spacing_count_2ch  = {"train": [], "dev": [], "test": []}  # Tuples with spaces in each dimension (h, w, depth)
spacing_count_4ch  = {"train": [], "dev": [], "test": []}  # Tuples with spaces in each dimension (h, w, depth)
shapes_count_2ch   = {"train": {}, "dev": {}, "test": {}}  # To store the different shapes in the data -> key: shape tuple string, value: number of adquisitions
shapes_count_4ch   = {"train": {}, "dev": {}, "test": {}}  # To store the different shapes in the data -> key: shape tuple string, value: number of adquisitions
pixels_stats_2ch   = {"train": [], "dev": [], "test": []}  # Tuples of pixels values (avg, max, min)
pixels_stats_4ch   = {"train": [], "dev": [], "test": []}  # Tuples of pixels values (avg, max, min)

for split_name, data_path in data_splits:
    for patient in tqdm(os.listdir(data_path), desc=f"{split_name + ' split':<13}"):
        sax_dirs = glob.glob(os.path.join(data_path, patient, "study/sax_*"))  # Get all the slices folders of type sax
        slices_count[split_name].append(len(sax_dirs))  # To count the number of slices per patient

        # Get the number of auxiliary views (2ch and 4ch)
        two_ch_dirs = glob.glob(os.path.join(data_path, patient, "study/2ch_*"))  
        four_ch_dirs = glob.glob(os.path.join(data_path, patient, "study/4ch_*"))
        two_ch_count[split_name].append(len(two_ch_dirs))
        four_ch_count[split_name].append(len(four_ch_dirs))

        has_2ch, has_4ch = False, False
        if len(two_ch_dirs) == 0 and len(four_ch_dirs) > 0:
            miss_ch_count[split_name].append(1)  # 1 for missing 2CH
            has_4ch = True
        elif len(two_ch_dirs) > 0 and len(four_ch_dirs) == 0:
            miss_ch_count[split_name].append(2)  # 2 for missing 4CH
            has_2ch = True
        elif len(two_ch_dirs) == 0 and len(four_ch_dirs) == 0:
            miss_ch_count[split_name].append(3)  # 3 for missing 2CH and 4CH
        else:
            has_2ch = True
            has_4ch = True

        '''Analyze 2CH views'''
        for two_ch_dir in two_ch_dirs:
            # Store the original number of frames
            n_frames = len([f_name for f_name in os.listdir(two_ch_dir) if f_name.endswith(".dcm")])
            frames_count_2ch[split_name].append(n_frames)

            dicom_files = get_dicom_files(two_ch_dir)  # Get the list of valid dicom files in the 2ch folder

            first = True  # Take some data just from the first frame of the slice

            for dicom_name in dicom_files:
                if dicom_name == "<BLACK>":
                    ''' This is a black synthetic slice created for the slices
                    with less than 30 frames in the slive video'''
                    continue 
                else:
                    dicom_data = dcmread(os.path.join(two_ch_dir, dicom_name)) 
                    pixel_array = dicom_data.pixel_array
                    if first:
                        first = False
                        # Get the spacing between pixels to scale the images
                        x_dist, y_dist = dicom_data.PixelSpacing  # Row and column spacing
                        x_dist, y_dist = float(x_dist), float(y_dist)
                        z_dist = float(dicom_data.SliceThickness)  # Depth spacing
                        spacing_count_2ch[split_name].append((x_dist, y_dist, z_dist))
                        # Store slices shape
                        img_shape = pixel_array.shape
                        shapes_count_2ch[split_name][str(img_shape)] = shapes_count[split_name].get(str(img_shape), 0) + 1  

                    # Get pixel values stats
                    pix_avg = np.mean(pixel_array)
                    pix_max = np.max(pixel_array)
                    pix_min = np.min(pixel_array)
                    pixels_stats_2ch[split_name].append((pix_avg, pix_max, pix_min))

        '''Analyze 4CH views'''
        for four_ch_dir in four_ch_dirs:
            # Store the original number of frames
            n_frames = len([f_name for f_name in os.listdir(four_ch_dir) if f_name.endswith(".dcm")])
            frames_count_4ch[split_name].append(n_frames)

            dicom_files = get_dicom_files(four_ch_dir)  # Get the list of valid dicom files in the 2ch folder

            first = True  # Take some data just from the first frame of the slice

            for dicom_name in dicom_files:
                if dicom_name == "<BLACK>":
                    ''' This is a black synthetic slice created for the slices
                    with less than 30 frames in the slive video'''
                    continue 
                else:
                    dicom_data = dcmread(os.path.join(four_ch_dir, dicom_name)) 
                    pixel_array = dicom_data.pixel_array
                    if first:
                        first = False
                        # Get the spacing between pixels to scale the images
                        x_dist, y_dist = dicom_data.PixelSpacing  # Row and column spacing
                        x_dist, y_dist = float(x_dist), float(y_dist)
                        z_dist = float(dicom_data.SliceThickness)  # Depth spacing
                        spacing_count_4ch[split_name].append((x_dist, y_dist, z_dist))
                        # Store slices shape
                        img_shape = pixel_array.shape
                        shapes_count_4ch[split_name][str(img_shape)] = shapes_count[split_name].get(str(img_shape), 0) + 1  

                    # Get pixel values stats
                    pix_avg = np.mean(pixel_array)
                    pix_max = np.max(pixel_array)
                    pix_min = np.min(pixel_array)
                    pixels_stats_4ch[split_name].append((pix_avg, pix_max, pix_min))

        '''Analyze SAX views'''
        for sax_dir in sax_dirs:
            # Store the original number of frames
            n_frames = len([f_name for f_name in os.listdir(sax_dir) if f_name.endswith(".dcm")])
            frames_count[split_name].append(n_frames)

            dicom_files = get_dicom_files(sax_dir)  # Get the list of valid dicom files in the sax folder

            first = True  # Take some data just from the first frame of the slice

            for dicom_name in dicom_files:
                if dicom_name == "<BLACK>":
                    ''' This is a black synthetic slice created for the slices
                    with less than 30 frames in the slive video'''
                    continue 
                else:
                    dicom_data = dcmread(os.path.join(sax_dir, dicom_name)) 
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
    print(f"{split}: mean={mean(counts):.2f}, median={median(counts):.2f}, mode={mode(counts):.2f}, max={max(counts)}, min={min(counts)}")

# Histogram of slices counts
plt.hist([x[1] for x in slices_count.items()], bins=30, label=[x[0] for x in slices_count.items()])
plt.legend(loc="upper right")
plt.xlabel("N slices")
plt.ylabel("Count")
plt.title(f"Count of slices by patient for each partition")
plt.savefig(os.path.join(out_path, f"slices_count.png"))
plt.clf()  # Reset figure for next plot

print("\nSlices count by patient (log scale):")

# Histogram of slices counts
plt.hist([x[1] for x in slices_count.items()], bins=30, label=[x[0] for x in slices_count.items()])
plt.legend(loc="upper right")
plt.xlabel("N slices")
plt.ylabel("Count (log scale)")
plt.title(f"Count of slices by patient for each partition")
plt.yscale('log', nonpositive='clip')
plt.savefig(os.path.join(out_path, f"log_slices_count.png"))
plt.clf()  # Reset figure for next plot

'''
Auxiliary views stats
'''
print("\n2CH views by patient:")
for split, counts in two_ch_count.items():
    print(f"{split}: mean={mean(counts):.2f}, median={median(counts):.2f}, mode={mode(counts):.2f}, max={max(counts)}, min={min(counts)}")

# Histogram of 2ch counts
plt.xlim(-0.25, 1.25)
plt.xticks([0, 1])
plt.hist([x[1] for x in two_ch_count.items()], bins=5, label=[x[0] for x in two_ch_count.items()])
plt.legend(loc="upper left")
plt.xlabel("N slices")
plt.ylabel("Count")
plt.title(f"Count of 2CH views by patient for each partition")
plt.savefig(os.path.join(out_path, f"2ch_count.png"))
plt.clf()  # Reset figure for next plot

# Histogram of 2ch counts (log scale)
plt.xlim(-0.25, 1.25)
plt.xticks([0, 1])
plt.hist([x[1] for x in two_ch_count.items()], bins=5, label=[x[0] for x in two_ch_count.items()])
plt.legend(loc="upper left")
plt.xlabel("N slices")
plt.ylabel("Count (log scale)")
plt.title(f"Count of 2CH views by patient for each partition")
plt.yscale('log', nonpositive='clip')
plt.savefig(os.path.join(out_path, f"log_2ch_count.png"))
plt.clf()  # Reset figure for next plot

print("\n4CH views by patient:")
for split, counts in four_ch_count.items():
    print(f"{split}: mean={mean(counts):.2f}, median={median(counts):.2f}, mode={mode(counts):.2f}, max={max(counts)}, min={min(counts)}")

# Histogram of 4ch counts
plt.xlim(-0.25, 1.25)
plt.xticks([0, 1])
plt.hist([x[1] for x in four_ch_count.items()], bins=5, label=[x[0] for x in four_ch_count.items()])
plt.legend(loc="upper left")
plt.xlabel("N slices")
plt.ylabel("Count")
plt.title(f"Count of 4CH views by patient for each partition")
plt.savefig(os.path.join(out_path, f"4ch_count.png"))
plt.clf()  # Reset figure for next plot

# Histogram of 4ch counts (log scale)
plt.xlim(-0.25, 1.25)
plt.xticks([0, 1])
plt.hist([x[1] for x in four_ch_count.items()], bins=5, label=[x[0] for x in four_ch_count.items()])
plt.legend(loc="upper left")
plt.xlabel("N slices")
plt.ylabel("Count (log scale)")
plt.title(f"Count of 4CH views by patient for each partition")
plt.yscale('log', nonpositive='clip')
plt.savefig(os.path.join(out_path, f"log_4ch_count.png"))
plt.clf()  # Reset figure for next plot

# Histogram of different cases of missing CH views
plt.xticks([1, 2, 3], ["Only 4CH", "Only 2CH", "No CH"])
plt.hist([x[1] for x in miss_ch_count.items()], bins=10, label=[x[0] for x in miss_ch_count.items()])
plt.legend(loc="upper right")
plt.ylabel("Count")
plt.title(f"Count of cases for missing CH views for each partition")
plt.savefig(os.path.join(out_path, f"missing_ch_count.png"))
plt.clf()  # Reset figure for next plot

'''
Frames stats
'''

print("\nSAX frames count by slice:")
for split, counts in frames_count.items():
    print(f"{split}: mean={mean(counts):.2f}, median={median(counts):.2f}, mode={mode(counts):.2f}, max={max(counts)}, min={min(counts)}")

# Histogram of frame counts
plt.hist([x[1] for x in frames_count.items()], bins=20, label=[x[0] for x in frames_count.items()])
plt.legend(loc="upper right")
plt.xlabel("N frames")
plt.ylabel("Count")
plt.title(f"Count of frames by slice for each partition")
plt.savefig(os.path.join(out_path, f"frames_count.png"))
plt.clf()  # Reset figure for next plot

# Histogram of frame counts (log scale)
plt.hist([x[1] for x in frames_count.items()], bins=20, label=[x[0] for x in frames_count.items()])
plt.legend(loc="upper right")
plt.xlabel("N frames")
plt.ylabel("Count (log scale)")
plt.title(f"Count of frames by SAX slice for each partition")
plt.yscale('log', nonpositive='clip')
plt.savefig(os.path.join(out_path, f"log_frames_count.png"))
plt.clf()  # Reset figure for next plot

print("\n2CH frames count by slice:")
for split, counts in frames_count_2ch.items():
    print(f"{split}: mean={mean(counts):.2f}, median={median(counts):.2f}, mode={mode(counts):.2f}, max={max(counts)}, min={min(counts)}")

# Histogram of frame counts
plt.hist([x[1] for x in frames_count_2ch.items()], bins=20, label=[x[0] for x in frames_count_2ch.items()])
plt.legend(loc="upper right")
plt.xlabel("N frames")
plt.ylabel("Count")
plt.title(f"Count of frames by 2CH slice for each partition")
plt.savefig(os.path.join(out_path, f"frames_count_2ch.png"))
plt.clf()  # Reset figure for next plot

# Histogram of frame counts (log scale)
plt.hist([x[1] for x in frames_count_2ch.items()], bins=20, label=[x[0] for x in frames_count_2ch.items()])
plt.legend(loc="upper right")
plt.xlabel("N frames")
plt.ylabel("Count (log scale)")
plt.title(f"Count of frames by 2CH slice for each partition")
plt.yscale('log', nonpositive='clip')
plt.savefig(os.path.join(out_path, f"log_frames_count_2ch.png"))
plt.clf()  # Reset figure for next plot

print("\n4CH frames count by slice:")
for split, counts in frames_count_4ch.items():
    print(f"{split}: mean={mean(counts):.2f}, median={median(counts):.2f}, mode={mode(counts):.2f}, max={max(counts)}, min={min(counts)}")

# Histogram of frame counts
plt.hist([x[1] for x in frames_count_4ch.items()], bins=20, label=[x[0] for x in frames_count_4ch.items()])
plt.legend(loc="upper right")
plt.xlabel("N frames")
plt.ylabel("Count")
plt.title(f"Count of frames by 4CH slice for each partition")
plt.savefig(os.path.join(out_path, f"frames_count_4ch.png"))
plt.clf()  # Reset figure for next plot

# Histogram of frame counts (log scale)
plt.hist([x[1] for x in frames_count_4ch.items()], bins=20, label=[x[0] for x in frames_count_4ch.items()])
plt.legend(loc="upper right")
plt.xlabel("N frames")
plt.ylabel("Count (log scale)")
plt.title(f"Count of frames by 4CH slice for each partition")
plt.yscale('log', nonpositive='clip')
plt.savefig(os.path.join(out_path, f"log_frames_count_4ch.png"))
plt.clf()  # Reset figure for next plot

'''
Shapes stats
'''
print("\nSAX shapes:")
for split, shapes in shapes_count.items():
    print(f"{split} split:")
    for shape, count in sorted(shapes.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{shape}: {count:>4}")

print("For all the splits:")
# Merge splits dicts
all_shapes_count = {}
for split, shapes in shapes_count.items():
    for shape, count in shapes.items():
        all_shapes_count[shape] = all_shapes_count.get(shape, 0) + count
# Print the shapes count
for shape, count in sorted(all_shapes_count.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{shape}: {count:>4}")

print("\n2CH shapes:")
for split, shapes in shapes_count_2ch.items():
    print(f"{split} split:")
    for shape, count in sorted(shapes.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{shape}: {count:>4}")

print("For all the splits:")
# Merge splits dicts
all_shapes_count_2ch = {}
for split, shapes in shapes_count_2ch.items():
    for shape, count in shapes.items():
        all_shapes_count_2ch[shape] = all_shapes_count_2ch.get(shape, 0) + count
# Print the shapes count
for shape, count in sorted(all_shapes_count_2ch.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{shape}: {count:>4}")

print("\n4CH shapes:")
for split, shapes in shapes_count_4ch.items():
    print(f"{split} split:")
    for shape, count in sorted(shapes.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{shape}: {count:>4}")

print("For all the splits:")
# Merge splits dicts
all_shapes_count_4ch = {}
for split, shapes in shapes_count_4ch.items():
    for shape, count in shapes.items():
        all_shapes_count_4ch[shape] = all_shapes_count_4ch.get(shape, 0) + count
# Print the shapes count
for shape, count in sorted(all_shapes_count_4ch.items(), key=lambda x: x[1], reverse=True):
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
plt.savefig(os.path.join(out_path, f"spacings_x.png"))
plt.clf()  # Reset figure for next plot

# Column spacings histogram
plt.hist(y_dists, bins=30)
plt.xlabel("Distance")
plt.ylabel("Count")
plt.title(f"Count of column spacings")
plt.savefig(os.path.join(out_path, f"spacings_y.png"))
plt.clf()  # Reset figure for next plot

# Depth spacings histogram
plt.hist(z_dists, bins=30)
plt.xlabel("Distance")
plt.ylabel("Count")
plt.title(f"Count of depth spacings")
plt.savefig(os.path.join(out_path, f"spacings_z.png"))
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
plt.savefig(os.path.join(out_path, f"pixels_averages.png"))
plt.clf()  # Reset figure for next plot

# Pixels maximums histogram
plt.hist(pix_maxs, bins=30)
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.title(f"Count of maximum pixel values for every dicom image")
plt.savefig(os.path.join(out_path, f"pixels_maxs.png"))
plt.clf()  # Reset figure for next plot

# Pixels minimums histogram
plt.hist(pix_mins, bins=30)
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.title(f"Count of minimum pixel values for every dicom image")
plt.savefig(os.path.join(out_path, f"pixels_mins.png"))
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
plt.savefig(os.path.join(out_path, f"labels_count.png"))
plt.clf()  # Reset figure for next plot
