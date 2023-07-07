from types import new_class
from torch.utils.data import Dataset
import torch
import os
import random
import math
import torchaudio

class YaseenDataset(Dataset):  #Inheritance
    __slots__ =('audio_attribute')
    def __init__(self, datafolder, filepath):
        # Filepath = os.path.join(datafolder, filepath)
        self.datafolder = datafolder
        with open(filepath) as fileobj:  #closed at the end
            self.audio_attribute= [line.strip() for line in fileobj]

        self.labels=['N', 'MS', 'MR', 'MVP', 'AS']

    def label_to_index(self, word):
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.audio_attribute);
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.datafolder+self.audio_attribute[idx], normalize=True)

        s=self.audio_attribute[idx]
        label=self.label_to_index(s.split('/')[-1].split('_')[-2])
        return (waveform,sample_rate, label)

class PascalDataset(Dataset):  #Inheritance
    __slots__ =('audio_attribute')
    def __init__(self, datafolder, filepath, subset):
        '''
        Subset refers to the subset of the subset of the PascalDataset to use
        options are 'A', 'B_clean' and 'B_noisy'
        '''
        # Filepath = os.path.join(datafolder, filepath)
        self.datafolder = datafolder
        with open(filepath) as fileobj:  #closed at the end
            self.audio_attribute= [line.strip() for line in fileobj]
        self.subset = subset

        if self.subset == 'A':
            self.label_cleaner= lambda s:s.split('training_')[1].split('/')[0]
        elif self.subset == 'B_clean':
            self.label_cleaner= lambda s:s.split('raining')[1].split('/')[0].strip("_").strip(" B ").lower()
        elif self.subset == 'B_noisy':
            self.label_cleaner= lambda s:s.split("noisy")[-1].split("/")[0]
        else:
            raise Exception("subset not implemented")

        self.labels=['normal', 'murmur']

    def label_to_index(self, word):
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.audio_attribute);
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.datafolder+self.audio_attribute[idx], normalize=True)

        s=self.audio_attribute[idx]        
        try:
            label=self.label_to_index(self.label_cleaner(s))
        except:
            print(s)
            raise Exception(s,self.label_cleaner(s))
        return (waveform,sample_rate, label)

if __name__ == '__main__':
    testset = YaseenDataset("/scratch/jiaqi006/others/Yaseen_CHSSUMF",'split_lists/testing_2.txt')
    print(testset[0][0].shape)

    data_dir = "/scratch/jiaqi006/others/PASCAL"
    dataA = PascalDataset(data_dir,"./pascal_lists/DatasetA_n_m.txt","A")
    print(dataA[0])
    dataBc = PascalDataset(data_dir,"./pascal_lists/DatasetB_clean_n_m.txt","B_clean")
    print(dataBc[0])
    dataBn = PascalDataset(data_dir,"./pascal_lists/DatasetB_noisy_n_m.txt","B_noisy")
    print(dataBn[0])
