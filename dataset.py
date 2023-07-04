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

if __name__ == '__main__':
    testset = YaseenDataset("/scratch/jiaqi006/others/Yaseen_CHSSUMF",'split_lists/testing_2.txt')
    print(testset[0][0].shape)
