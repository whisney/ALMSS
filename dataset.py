from torch.utils.data import Dataset
import os
import torch
from Nii_utils import NiiDataRead
import pandas as pd
from volumentations import *

class Dataset_for_primary_tumor(Dataset):
    def __init__(self, data_dir, split_path, metadata_path, augment=True):
        self.data_dir = data_dir
        self.augment = augment

        with open(split_path, 'r') as f:
            ID_list_orginal = f.readlines()
        ID_list_orginal = [n.strip('\n') for n in ID_list_orginal]

        metadata_df = pd.read_excel(metadata_path)
        metadata_df['ID'] = metadata_df['ID'].astype(str)
        metadata_df['label'] = metadata_df['label'].astype(int)

        self.ID_list = []
        self.gender_list = []
        self.age_list = []
        self.label_list = []

        for ID in ID_list_orginal:
            label_one = metadata_df.loc[metadata_df.ID == ID, 'label'].values[0]
            if label_one <= 6:
                self.ID_list.append(ID)
                self.label_list.append(0)
                self.gender_list.append(str(metadata_df.loc[metadata_df.ID == ID, 'sex'].values[0]))
                self.age_list.append(int(metadata_df.loc[metadata_df.ID == ID, 'age'].values[0]))
            else:
                self.ID_list.append(ID)
                self.label_list.append(label_one - 6)
                self.gender_list.append(str(metadata_df.loc[metadata_df.ID == ID, 'sex'].values[0]))
                self.age_list.append(int(metadata_df.loc[metadata_df.ID == ID, 'age'].values[0]))

        self.num_0 = self.label_list.count(0)
        self.num_1 = self.label_list.count(1)
        self.num_2 = self.label_list.count(2)

        self.transforms = Compose([
            RotatePseudo2D(axes=(1, 2), limit=(-30, 30), interpolation=3, value=-1, p=0.3),
            ElasticTransformPseudo2D(alpha=50, sigma=30, alpha_affine=10, value=-1, p=0.3),
            GaussianNoise(var_limit=(0, 0.1), mean=0, p=0.3),
        ])

        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        gender = self.gender_list[idx]
        age = self.age_list[idx]
        label = self.label_list[idx]

        if 'male' in gender:
            gender = torch.from_numpy(np.array([1, 0]))
        elif 'female' in gender:
            gender = torch.from_numpy(np.array([0, 1]))

        age = torch.from_numpy(np.array([age])).float() / 100

        Plain_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Plain_img_liver.nii.gz'))
        Arterial_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Arterial_img_liver.nii.gz'))
        Venous_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Venous_img_liver.nii.gz'))
        Delay_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Delay_img_liver.nii.gz'))

        Plain_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Plain_img_tumor.nii.gz'))
        Arterial_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Arterial_img_tumor.nii.gz'))
        Venous_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Venous_img_tumor.nii.gz'))
        Delay_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Delay_img_tumor.nii.gz'))
        img = np.concatenate((Plain_img_liver[..., np.newaxis], Plain_img_tumor[..., np.newaxis],
                              Arterial_img_liver[..., np.newaxis], Arterial_img_tumor[..., np.newaxis],
                              Venous_img_liver[..., np.newaxis], Venous_img_tumor[..., np.newaxis],
                              Delay_img_liver[..., np.newaxis], Delay_img_tumor[..., np.newaxis]), axis=-1)

        if self.augment:
            img = self.transforms(image=img)['image']
            age += (torch.rand(*age.size()) - 0.5) * 0.04

        img = torch.from_numpy(img).permute(3, 0, 1, 2)

        Plain_img = img[0:2]
        Arterial_img = img[2:4]
        Venous_img = img[4:6]
        Delay_img = img[6:8]
        gender_age = torch.cat((gender, age))
        label = torch.tensor(label)
        return Plain_img, Arterial_img, Venous_img, Delay_img, gender_age, label

    def __len__(self):
        return self.len

class Dataset_for_metastatic_tumor(Dataset):
    def __init__(self, data_dir, split_path, metadata_path, augment=True):
        self.data_dir = data_dir
        self.augment = augment

        with open(split_path, 'r') as f:
            ID_list_orginal = f.readlines()
        ID_list_orginal = [n.strip('\n') for n in ID_list_orginal]

        metadata_df = pd.read_excel(metadata_path)
        metadata_df['ID'] = metadata_df['ID'].astype(str)
        metadata_df['label'] = metadata_df['label'].astype(int)

        self.ID_list = []
        self.gender_list = []
        self.age_list = []
        self.label_list = []

        for ID in ID_list_orginal:
            label_one = metadata_df.loc[metadata_df.ID == ID, 'label'].values[0]
            if label_one <= 6:
                self.ID_list.append(ID)
                self.label_list.append(label_one - 1)
                self.gender_list.append(str(metadata_df.loc[metadata_df.ID == ID, 'sex'].values[0]))
                self.age_list.append(int(metadata_df.loc[metadata_df.ID == ID, 'age'].values[0]))

        self.num_0 = self.label_list.count(0)
        self.num_1 = self.label_list.count(1)
        self.num_2 = self.label_list.count(2)
        self.num_3 = self.label_list.count(3)
        self.num_4 = self.label_list.count(4)
        self.num_5 = self.label_list.count(5)

        self.transforms = Compose([
            RotatePseudo2D(axes=(1, 2), limit=(-30, 30), interpolation=3, value=-1, p=0.3),
            ElasticTransformPseudo2D(alpha=50, sigma=30, alpha_affine=10, value=-1, p=0.3),
            GaussianNoise(var_limit=(0, 0.1), mean=0, p=0.3),
        ])

        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        gender = self.gender_list[idx]
        age = self.age_list[idx]
        label = self.label_list[idx]

        if 'male' in gender:
            gender = torch.from_numpy(np.array([1, 0]))
        elif 'female' in gender:
            gender = torch.from_numpy(np.array([0, 1]))

        age = torch.from_numpy(np.array([age])).float() / 100

        Plain_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Plain_img_liver.nii.gz'))
        Arterial_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Arterial_img_liver.nii.gz'))
        Venous_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Venous_img_liver.nii.gz'))
        Delay_img_liver, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Delay_img_liver.nii.gz'))

        Plain_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Plain_img_tumor.nii.gz'))
        Arterial_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Arterial_img_tumor.nii.gz'))
        Venous_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Venous_img_tumor.nii.gz'))
        Delay_img_tumor, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, 'Delay_img_tumor.nii.gz'))
        img = np.concatenate((Plain_img_liver[..., np.newaxis], Plain_img_tumor[..., np.newaxis],
                              Arterial_img_liver[..., np.newaxis], Arterial_img_tumor[..., np.newaxis],
                              Venous_img_liver[..., np.newaxis], Venous_img_tumor[..., np.newaxis],
                              Delay_img_liver[..., np.newaxis], Delay_img_tumor[..., np.newaxis]), axis=-1)

        if self.augment:
            img = self.transforms(image=img)['image']
            age += (torch.rand(*age.size()) - 0.5) * 0.04

        img = torch.from_numpy(img).permute(3, 0, 1, 2)

        Plain_img = img[0:2]
        Arterial_img = img[2:4]
        Venous_img = img[4:6]
        Delay_img = img[6:8]
        gender_age = torch.cat((gender, age))
        label = torch.tensor(label)
        return Plain_img, Arterial_img, Venous_img, Delay_img, gender_age, label

    def __len__(self):
        return self.len