import os

import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

dataname = {
    0: ["Normal_0.mat", "IR007_0.mat", "B007_0.mat", "OR007@6_0.mat", "IR014_0.mat", "B014_0.mat", "OR014@6_0.mat",
        "IR021_0.mat", "B021_0.mat", "OR021@6_0.mat"],  # 1797rpm
    1: ["Normal_1.mat", "IR007_1.mat", "B007_1.mat", "OR007@6_1.mat", "IR014_1.mat", "B014_1.mat", "OR014@6_1.mat",
        "IR021_1.mat", "B021_1.mat", "OR021@6_1.mat"],  # 1772rpm
    2: ["Normal_2.mat", "IR007_2.mat", "B007_2.mat", "OR007@6_2.mat", "IR014_2.mat", "B014_2.mat", "OR014@6_2.mat",
        "IR021_2.mat", "B021_2.mat", "OR021@6_2.mat"],  # 1750rpm
    3: ["Normal_3.mat", "IR007_3.mat", "B007_3.mat", "OR007@6_3.mat", "IR014_3.mat", "B014_3.mat", "OR014@6_3.mat",
        "IR021_3.mat", "B021_3.mat", "OR021@6_3.mat"]}  # 1730rpm

label = [i for i in range(0, len(dataname[0]))]

# Fs = 12kHz
signal_size = 1024


#  --------------------------------------------------------------------------------------------------------------------
def get_files(root, N):
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def data_load(filename, label):
    mat_data = loadmat(filename)

    variable_names = mat_data.keys()

    filtered_variables = {var_name: mat_data[var_name] for var_name in variable_names if "_DE_time" in var_name}

    for var_name, var_value in filtered_variables.items():
        fl = var_value
        # print("========================")
        # print(var_name)
        # print("========================")

    data = []
    lab = []

    start, end = 0, signal_size
    step_size = signal_size // 4

    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += step_size
        end = start + signal_size

    return data, lab


#  --------------------------------------------------------------------------------------------------------------------
class CWRU(object):

    def __init__(self, data_dir, transfer_task, normlizetype):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                RandomAddGaussian(),
                RandomScale(),
                RandomStretch(),
                RandomCrop(),
                Retype(),
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ])
        }

    def data_split(self, sample_number=256, Data_dependency=False):
        # get source train and val
        list_data = get_files(self.data_dir, self.source_N)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        label_counts = data_pd['label'].value_counts()
        print(f"The number of samples for each category in the source domain dataset is：\n{label_counts}")
        sampled_data_pd = data_pd.groupby('label').apply(
            lambda x: x.sample(n=sample_number)).reset_index(drop=True)
        train_pd, val_pd = train_test_split(sampled_data_pd, test_size=0.3,
                                            stratify=sampled_data_pd["label"])
        source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
        source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

        # get target train and val
        list_data = get_files(self.data_dir, self.target_N)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        label_counts = data_pd['label'].value_counts()
        print(f"The number of samples for each category in the target domain dataset is：\n{label_counts}")
        sampled_data_pd = data_pd.groupby('label').apply(
            lambda x: x.sample(n=sample_number)).reset_index(drop=True)

        if Data_dependency:
            train_pd_per, val_pd = train_test_split(sampled_data_pd, test_size=0.3,
                                                stratify=sampled_data_pd["label"])
            train_pd,  train_pd_dis = train_test_split(train_pd_per, test_size=(1-1/7),
                                                    stratify=train_pd_per["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
        else:
            train_pd, val_pd = train_test_split(sampled_data_pd, test_size=0.3,
                                                stratify=sampled_data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])



        return source_train, source_val, target_train, target_val
