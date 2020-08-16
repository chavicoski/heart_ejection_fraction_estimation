import sys
sys.path.insert(1, '.')  # To access the libraries
import torch
from lib.my_transforms import MyAffine, ChannelShift
from torchvision import transforms

class Cardiac_dataset(torch.utils.data.Dataset):
    def __init__(self, data_df, target_label, data_augmentation=0, view="SAX"):
        '''Dataset constructor
        Params:
            data_df -> Pandas dataframe with the dataset info
                *data_df columns:
                    - Id -> case Id in the dataset
                    - View -> "SAX", "2CH" or "4CH"
                    - SliceId -> Id of the slice inside the case
                    - X_path -> relative path from the workspace folder to the data tensor
                    - Systole -> float value for Systole label
                    - Diastole -> float value for Diastole label
            target_label -> Label to return, "Sytole" or "Diastole"
            data_augmentation ->    0: Disables DA
                                    1: Enables DA
                                    2: Add ChannelShift to DA
                                    3: Like the option 2 but more intense
            view -> To select a view for the generator. choices=["SAX", "2CH", "4CH"]
        '''
        if view not in ["SAX", "2CH", "4CH"]:
            print(f"View {view} is not valid!")
            sys.exit()
        else:
            self.df = data_df[data_df["View"]==view]

        if target_label in ["Systole", "Diastole"]:
            self.target_label = target_label
        else:
            print(f"Wrong target label ({target_label}) passed to the dataset!")
            sys.exit()

        self.da = data_augmentation
        if self.da == 1:
            self.transform = MyAffine(angle_range=(-15, 15), translate_range=(0, 0.1))
        elif self.da == 2:
            self.transform = transforms.Compose([
                    MyAffine(angle_range=(-15, 15), translate_range=(0, 0.1)),
                    ChannelShift(shift_range=[-0.4, 0.4])
                ])
        elif self.da == 3:
            self.transform = transforms.Compose([
                    MyAffine(angle_range=(-30, 30), translate_range=(-0.1, 0.1)),
                    ChannelShift(shift_range=[-0.6, 0.6])
                ])

    def __len__(self):
        '''Returns the number of samples in the dataset'''
        return len(self.df.index)  # Number of rows in the DataFrame

    def __getitem__(self, index):
        '''Returns a sample by index from the dataset'''
        sample_row = self.df.iloc[index]  # Get sample row from DataFrame
        sample_data = torch.load(sample_row["X_path"])  # Load sample data
        if self.da:
            sample_data = self.transform(sample_data)  # Apply data augmentation
        sample_label = torch.tensor([sample_row[self.target_label]]).float()
        return {"X": sample_data, "Y": sample_label}
    

class Submission_dataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        '''Dataset constructor
        This dataset resturns all the samples for each patient Id
        Params:
            data_df -> Pandas dataframe with the dataset info
                *data_df columns:
                    - Id -> case Id in the dataset
                    - View -> "SAX", "2CH" or "4CH"
                    - SliceId -> Id of the slice inside the case
                    - X_path -> relative path from the workspace folder to the data tensor
                    - Systole -> float value for Systole label
                    - Diastole -> float value for Diastole label
        '''
        self.df = data_df
        self.ids = data_df["Id"].unique()

    def __len__(self):
        '''Returns the number of samples in the dataset'''
        return len(self.ids)  # Number of patients in the dataframe

    def __getitem__(self, index):
        '''Returns a sample by index from the dataset'''
        case_id = self.ids[index]  # Get the corresponding patient id
        selected_rows = self.df[self.df["Id"] == case_id]  # Get samples of the selected patient

        '''Get SAX slices'''
        slices = []
        sax_rows = selected_rows[selected_rows["View"]=="SAX"]
        for idx, row in sax_rows.iterrows():
            slices.append(torch.load(row["X_path"]))  # Load sample data
        case_data = torch.stack(slices)

        '''Get CH slices'''
        rows_2ch = selected_rows[selected_rows["View"]=="2CH"]
        rows_4ch = selected_rows[selected_rows["View"]=="4CH"]
        if len(rows_2ch.index) > 0:
            row = rows_2ch.iloc[0]
            data_2ch = torch.load(row["X_path"])
        else:
            data_2ch = None

        if len(rows_4ch.index) > 0:
            row = rows_4ch.iloc[0]
            data_4ch = torch.load(row["X_path"])
        else:
            data_4ch = None

        label_systole = torch.tensor([selected_rows.iloc[0]["Systole"]]).float()
        label_diastole = torch.tensor([selected_rows.iloc[0]["Diastole"]]).float()
        return {"ID": case_id, "X": case_data, "2CH": data_2ch, "4CH": data_4ch, "Y_systole": label_systole, "Y_diastole": label_diastole}


if __name__ == "__main__":
    
    import pandas as pd

    partition_csv_path = "../preproc1_150x150_bySlices_dataset/train.csv"
    target_label = "Systole"
    print(f"Creating dataset from: {partition_csv_path}")
    df = pd.read_csv(partition_csv_path)  # Load partition info
    dataset = Cardiac_dataset(df, target_label, data_augmentation=1)  # Create the dataset for the partition
    samples_to_print = 5
    print("TESTING Cardiac_dataset class:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Going to show {samples_to_print} samples:")
    for i, sample in enumerate(dataset):
        if i == samples_to_print: break
        x, y = sample['X'], sample['Y']
        print(f"\tX info: shape={x.shape} - mean={x.mean():.3f} - max={x.max()} - min={x.min()} - dtype={x.type()}")
        print(f"\tY info: {target_label}={y[0]:.3f} - dtype={y.type()}\n")


    print("TESTING Submission_dataset class:")
    dataset = Submission_dataset(df)  # Create the dataset for the partition
    samples_to_print = 5
    print(f"Number of samples: {len(dataset)}")
    print(f"Going to show {samples_to_print} samples:")
    for i, sample in enumerate(dataset):
        if i == samples_to_print: break
        id_, x, y_sys, y_dias = sample["ID"], sample['X'], sample['Y_systole'], sample["Y_diastole"]
        print(f"\tSample Id: {id_}")
        print(f"\tX info: shape={x.shape} - mean={x.mean():.3f} - max={x.max()} - min={x.min()} - dtype={x.type()}")
        print(f"\tY info: systole={y_sys[0]:.3f} - diastole={y_dias[0]:.3f} - dtype={y.type()}\n")
