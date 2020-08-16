import sys
import os
import argparse
sys.path.insert(1, '.')  # To access the libraries
from lib.image_processing import *

arg_parser = argparse.ArgumentParser(description="Creates the images for each time step of the preprocessing pipeline 1")
arg_parser.add_argument("case_number", help="The ID of the case to process", type=int)
arg_parser.add_argument("--split", help="The data split where the case belongs", type=str, choices=["train", "validate", "test"], default="train")
arg_parser.add_argument("--out_path", help="Path to the folder to store the plots", type=str, default="plots/images/")
arg_parser.add_argument("-th", "--target_height", help="Height of the processed images", type=int, default=150)
arg_parser.add_argument("-tw", "--target_width", help="Width of the processed images", type=int, default=150)
args = arg_parser.parse_args()

case_number = args.case_number
split = args.split
target_size = (args.target_height, args.target_width)
plots_path = os.path.join(args.out_path, f"case_{case_number}_preproc_steps")

patient_slices, pix_spacings = get_patient_slices(case_number, split)
preproc_patient = preprocess_pipeline1(patient_slices, pix_spacings, target_size=target_size, plots_path=plots_path) # Do preprocesing
