from types import new_class
from torch.utils.data import Dataset
import torch
import os
import random
import math
import torchaudio

class YaseenDataset(Dataset):  #Inheritance
    __slots__ =('audio_attribute')
    def __init__(self, datafolder, filepath, augment=False, include_orig=True):
        # Filepath = os.path.join(datafolder, filepath)
        self.datafolder = datafolder
        
        
        if augment:
            with open(filepath) as fileobj:  #closed at the end
                self.audio_attribute= [line.strip() for line in fileobj]
            
            temp_attributes = []
            if include_orig:
                aug_list = ['original','ogg4.5k','ogg5.5k','ogg7.7k']
            else:
                aug_list = ['ogg4.5k','ogg5.5k','ogg7.7k']
            for i in self.audio_attribute:
                for j in aug_list:
                    temp_attributes.append("/"+j+i)
            self.audio_attribute = temp_attributes
        else:
            with open(filepath) as fileobj:  #closed at the end
                self.audio_attribute= ["/original"+line.strip() for line in fileobj]
        
        # print(self.audio_attribute[0])
        # print(self.audio_attribute[-1])
        # print(len(self.audio_attribute))
        # raise Exception

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
        return (waveform, sample_rate, label)

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
        return len(self.audio_attribute)
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.datafolder+self.audio_attribute[idx], normalize=True)

        s=self.audio_attribute[idx]        
        try:
            label=self.label_to_index(self.label_cleaner(s))
        except:
            raise Exception(s,self.label_cleaner(s))
        return (waveform,sample_rate, label)

if __name__ == '__main__':
    testset = YaseenDataset("/scratch/jiaqi006/others/Yaseen_CHSSUMF",'split_lists/testing_2.txt',False)
    print(testset[0][1])
    print(testset[0][0].shape)

    data_dir = "/scratch/jiaqi006/others/PASCAL"
    dataA = PascalDataset(data_dir,"./pascal_lists/DatasetA_n_m.txt","A")
    print(dataA[0][2])
    print(dataA[0][1])
    print(dataA[0][0].shape)
    dataBc = PascalDataset(data_dir,"./pascal_lists/DatasetB_clean_n_m.txt","B_clean")
    print(dataBc[0][2])
    print(dataBc[0][1])
    print(dataBc[0][0].shape)
    dataBn = PascalDataset(data_dir,"./pascal_lists/DatasetB_noisy_n_m.txt","B_noisy")
    print(dataBn[0][2])
    print(dataBn[0][1])
    print(dataBn[0][0].shape)
