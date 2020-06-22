import sys
import torch

class Cardiac_dataset(torch.utils.data.Dataset):
    def __init__(self, data_df, target_label, norm_labels=False):
        '''Dataset constructor
        Params:
            data_df -> Pandas dataframe with the dataset info
                *data_df columns:
                    - Id -> case Id in the dataset
                    - X_path -> relative path from the workspace folder to the data tensor
                    - Systole -> float value for Systole label
                    - Diastole -> float value for Diastole label
            target_label -> Label to return, "Sytole" or "Diastole"
            norm_labels -> To enable the labels normalization between 0 and 1
        '''
        self.df = data_df
        self.norm_labels = norm_labels
        if target_label in ["Systole", "Diastole"]:
            self.target_label = target_label
        else:
            print(f"Wrong target label ({target_label}) passed to the dataset!")
            sys.exit()

    def __len__(self):
        '''Returns the number of samples in the dataset'''
        return len(self.df.index)  # Number of rows in the DataFrame

    def __getitem__(self, index):
        '''Returns a sample by index from the dataset'''
        sample_row = self.df.iloc[index]  # Get sample row from DataFrame
        sample_data = torch.load(sample_row["X_path"])  # Load sample data
        sample_label = torch.tensor([sample_row[self.target_label]]).float()
        return {"X": sample_data, "Y": sample_label}
    

if __name__ == "__main__":
    
    import pandas as pd

    partition_csv_path = "../preproc1_150x150_bySlices_dataset/train.csv"
    target_label = "Systole"
    print(f"Creating dataset from: {partition_csv_path}")
    df = pd.read_csv(partition_csv_path)  # Load partition info
    dataset = Cardiac_dataset(df, target_label)  # Create the dataset for the partition
    samples_to_print = 5
    print(f"Number of samples: {len(dataset)}")
    print(f"Going to show {samples_to_print} samples:")
    for i, sample in enumerate(dataset):
        if i == samples_to_print: break
        x, y = sample['X'], sample['Y']
        print(f"\tX info: shape={x.shape} - mean={x.mean():.3f} - max={x.max()} - min={x.min()} - dtype={x.type()}")
        print(f"\tY info: {target_label}={y[0]:.3f} - dtype={y.type()}\n")
