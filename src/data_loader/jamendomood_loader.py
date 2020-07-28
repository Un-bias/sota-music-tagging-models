# coding: utf-8
import pickle
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer

META_PATH = '/home/jupyter/models/sota/split/mtg-jamendo-mood'

TAGS = ["mood/theme---action", "mood/theme---adventure", "mood/theme---advertising", "mood/theme---background", "mood/theme---ballad", "mood/theme---calm", "mood/theme---children", "mood/theme---christmas", "mood/theme---commercial", "mood/theme---cool", "mood/theme---corporate", "mood/theme---dark", "mood/theme---deep", "mood/theme---documentary", "mood/theme---drama", "mood/theme---dramatic", "mood/theme---dream", "mood/theme---emotional", "mood/theme---energetic", "mood/theme---epic", "mood/theme---fast", "mood/theme---film", "mood/theme---fun", "mood/theme---funny", "mood/theme---game", "mood/theme---groovy", "mood/theme---happy", "mood/theme---heavy", "mood/theme---holiday", "mood/theme---hopeful", "mood/theme---inspiring", "mood/theme---love", "mood/theme---meditative", "mood/theme---melancholic", "mood/theme---melodic", "mood/theme---motivational", "mood/theme---movie", "mood/theme---nature", "mood/theme---party", "mood/theme---positive", "mood/theme---powerful", "mood/theme---relaxing", "mood/theme---retro", "mood/theme---romantic", "mood/theme---sad", "mood/theme---sexy", "mood/theme---slow", "mood/theme---soft", "mood/theme---soundscape", "mood/theme---space", "mood/theme---sport", "mood/theme---summer", "mood/theme---trailer", "mood/theme---travel", "mood/theme---upbeat", "mood/theme---uplifting"]

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy'),
                'tags': row[5:],
            }
    return tracks


class AudioFolder(data.Dataset):
    def __init__(self, root, split, input_length=None):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist()

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        self.mlb = LabelBinarizer().fit(TAGS)
        if self.split == 'TRAIN':
            train_file = os.path.join(META_PATH, 'autotagging_moodtheme-train.tsv')
            self.file_dict = read_file(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'VALID':
            train_file = os.path.join(META_PATH,'autotagging_moodtheme-validation.tsv')
            self.file_dict= read_file(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'TEST':
            test_file = os.path.join(META_PATH, 'autotagging_moodtheme-test.tsv')
            self.file_dict= read_file(test_file)
            self.fl = list(self.file_dict.keys())
        else:
            print('Split should be one of [TRAIN, VALID, TEST]')


    def get_npy(self, index):
        jmid = self.fl[index]
        filename = self.file_dict[jmid]['path']
        npy_path = os.path.join(self.root, filename.split("/")[-1])
        npy = np.load(npy_path, mmap_mode='r')
        random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
        npy = np.array(npy[random_idx:random_idx+self.input_length])
        tag_binary = np.sum(self.mlb.transform(self.file_dict[jmid]['tags']), axis=0)
        return npy, tag_binary

    def __len__(self):
        return len(self.fl)

def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader

