# coding: utf-8
import pickle
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer

META_PATH = '/home/jupyter/models/sota/split/genres'

TAGS = ['DANCE',  'K-POP',  'ALTERNATIVE',  'HIP-HOP/RAP',  'POP',  'LATINO',  'LATIN POP',  'LATIN URBAN',  'FUNK PARTY',  'JAZZ',  'ELECTRONIC',  'R&B/SOUL',  'POP IN SPANISH',  'CONTEMPORARY JAZZ',  'TECHNO',  'TRANCE',  'COUNTRY',  'AMBIENT',  'ROCK',  'INDIE ROCK',  'INDIE POP',  'HOUSE',  'HIPHOP/RAP',  'AMERICANA',  'HIP HOP / RAP',  'TROPICAL MUSIC',  'INDIE-ROCK',  'INDIE-POP',  'R & B / SOUL',  'SOUL',  'ALTERNATIVE RAP',  'LOUNGE',  'INSTRUMENTAL',  'SINGER-SONGWRITER',  'PERFORMER / COMPOSER',  'WORLD',  'SOUNDTRACK',  'URBAN LATINO-AMERICANA',  'DANCE MUSIC',  "JUNGLE/DRUM'N'BASS",  'CLASSICAL',  'RELIGIOUS MUSIC',  'NEW WAVE',  'FUNK',  'HOLIDAY',  'CONTEMPORARY FOLK',  'LATIN JAZZ',  'LATIN MUSIC',  'METAL',  'BRAZILIAN',  'ELECTRONIC MUSIC',  'FOLK',  'TRADITIONAL FOLK',  'SINGER/SONGWRITER',  'ADULT ALTERNATIVE',  'IDM/EXPERIMENTAL',  'DUBSTEP',  'DOWNTEMPO',  'LATIN',  'ALTERNATIVE FOLK MUSIC',  'ELECTONIC MUSIC',  'WORLDWIDE',  'SERTANEJO',  'SOUNDTRACKS',  'MODERN ERA',  'MEXICAN MUSIC',  'IDM / EXPERIMENTAL',  'REGGAE',  'FRENCH POP',  'ROCK INDIE',  'NEW AGE',  'R&B/SAO LINGLE',  'URBAN LATIN',  'CONTEMPORARY R&B',  'CONTEMPORARY SINGER/SONGWRITER',  'NEO-SOUL',  'JAZZ CROSSOVER',  'HARD ROCK',  'SONG WRITERS',  'PUNK',  'VOCAL',  'ORIGINAL SOUND TRACK',  'WORLD MUSIC',  'POP INDIE',  'BIG BAND',  'AUTHOR-PERFORMER',  'SMOOTH JAZZ',  'MODERN DANCEHALL',  'CHRISTIAN & GOSPEL',  'CHRISTMAS',  'BASS',  'BREAKBEAT',  'AFRO-POP',  'INTERNATIONAL POP',  'AVANT-GARDE JAZZ',  'FITNESS & WORKOUT',  'CONTEMPORARY ERA',  'OLD SCHOOL RAP',  'FUSION',  'CONTEMPORARY COUNTRY',  'ALTERNATIVE FOLK',  'JAZZ VOCAL']

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy').replace('.m4a', '.npy'),
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
            train_file = os.path.join(META_PATH, 'train.tsv')
            self.file_dict = read_file(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'VALID':
            train_file = os.path.join(META_PATH,'validation.tsv')
            self.file_dict= read_file(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'TEST':
            test_file = os.path.join(META_PATH, 'test.tsv')
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

