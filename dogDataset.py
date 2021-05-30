from os.path import basename
import glob
import torch.utils.data as data
from torch.utils.data import DataLoader
import librosa
import csv
import numpy as np


class Dogdataset(data.Dataset):

    def __init__(self, sound_dir, label_dir='./dataset/meta_train.csv', is_train=False):
        super(Dogdataset, self).__init__()
        self.is_train = is_train
        path_pattern = sound_dir + '*.wav'
        self.files_list = glob.glob(path_pattern, recursive=False)
        self.label = any
        try:
            with open(label_dir, newline='') as csvfile:
                self.label = list(csv.reader(csvfile))
                del self.label[0]
        except:
            pass

    def __getitem__(self, index):
        input_file = self.files_list[index]  # input_file 是路徑
        audio, sampling_rate = librosa.load(input_file, sr=None)
        mfccs = librosa.feature.mfcc(audio, sampling_rate, n_mfcc=128)
        '''
        if False:
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                             sr=sampling_rate,
                                                             n_mels=256,
                                                             hop_length=128,
                                                             fmax=8000)
            audio = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)
        '''
        if (self.is_train):
            return np.mean(mfccs.T, axis=0), int(self.label[index][1])
        else:
            filename = basename(input_file)
            filename = filename[:-4]
            return np.mean(mfccs.T, axis=0), filename

    def __len__(self):
        return len(self.files_list)
