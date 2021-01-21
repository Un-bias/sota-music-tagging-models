# coding: utf-8
import pickle
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer

META_PATH = '/home/jupyter/models/sota/split/genres'

TAGS = ['ALTERNATIVE',  'K-POP',  'SPANISH POP',  'HIP-HOP/RAP',  'R&B/SOUL',  'DANCE',  'POP',  'LATIN URBAN',  'LATIN POP',  'INDIE POP',  'INDIE ROCK',  'ELECTRONIC',  'JAZZ',  'ROCK',  'COUNTRY',  'MEXICAN MUSIC',  'FUSION',  'FUNK',  'ALTERNATIVE RAP',  'SINGER/SONGWRITER',  'BRAZILIAN',  'ALTERNATIVE FOLK',  'SOUNDTRACK',  'LATINO',  'PUNK',  'TRANCE',  'FRENCH POP',  'ELECTONIC MUSIC',  'LATIN MUSIC',  'SOUNDTRACKS',  'EASY LISTENING',  'HOUSE',  'CONTEMPORARY R&B',  'TECHNO',  'METAL',  'JAZZ VOCAL',  'WORLD',  'CHRISTIAN & GOSPEL',  'SOUND BAND',  'BREAKBEAT',  'JAZZ CROSSOVER',  'SERTANEJO',  'HIP HOP / RAP',  'NEO-SOUL',  'CONTEMPORARY JAZZ',  'VOCAL',  'TRADITIONAL FOLK',  'SALSA AND TROPICAL',  'URBAN LATIN',  'DOWNTEMPO',  'INDIE-ROCK',  'LATIN',  'HARD ROCK',  'PERFORMER / COMPOSER',  'INSPIRATIONAL',  'REGGAE',  'BIG BAND',  'POP IN SPANISH',  'R & B / SOUL',  'ALTERNATIVE FOLK MUSIC',  'INDIE-POP',  'SINGER-SONGWRITER',  'POP INDIE',  'IDM/EXPERIMENTAL',  'MODERN ERA',  'INDEPENDENT ROCK',  'J-POP',  'WORLD MUSIC',  'FUNK PARTY',  'CHRISTMAS',  'HOLIDAY',  'INTERNATIONAL POP',  'CONTEMPORARY FOLK',  'NEW WAVE',  'AUTHOR-PERFORMER',  'HIPHOP/RAP',  'FITNESS & WORKOUT',  'HIPHOP',  'ELECTRONIC MUSIC',  'COLLEGE ROCK',  'INSTRUMENTAL',  'CHRISTIAN MUSIC',  'TROPICAL MUSIC',  'SEOUL',  'AVANT-GARDE JAZZ',  'R&B/SAO LINGLE',  "JUNGLE/DRUM'N'BASS",  'AMBIENT',  'FOLK',  "CHILDREN'S MUSIC",  'URBAN LATIN MUSIC',  'ORIGINAL SCORE',  'SMOOTH JAZZ',  'WORLDWIDE',  'VOCAL JAZZ',  'DANCE MUSIC',  'SOUL',  'NEOSOUL',  'GRUNGE',  'FESTIVAL',  'IT WAS MODERN',  'LATIN AMERICAN',  '=DANCE=',  'CLASSICAL',  'UNDERGROUND RAP',  'SONG WRITERS',  'PSYCHEDELIC',  'OLD SCHOOL RAP',  'BASS',  'ADULT ALTERNATIVE',  'OLDIES',  'GARAGE',  'LATIN AMERICAN URBAN MUSIC',  'MODERN DANCEHALL',  'LATINA',  'ROCK INDIE',  'DUB',  'ADULT CONTEMPORARY',  'TRADITIONAL FOLK MUSIC',  'INDIAN POP',  'SLOW SHOT',  'R&B CONTEMPORAIN',  'ROCK: INDEPENDENT',  'LATIN JAZZ',  'POP PUNK',  'DUBSTEP',  'LOCK',  'BRAZILIAN MUSIC',  'NEO SOUL',  'ELECTRONICALLY',  'CONTRIBUTING AUTHORS',  'CONTEMPORARY COUNTRY',  'NEW AGE',  'LOUNGE',  'ORIGINAL SOUND TRACK',  'AFRO-POP',  'AMERICANA',  'HIP HOP',  'INDUSTRIAL',  'GOTH ROCK',  'BLUES',  'LATIN RAP',  'GOSPEL',  'MUSIC',  'BLUES-ROCK',  'IDM / EXPERIMENTAL',  'POP/ROCK',  'VOCAL MUSIC',  '=MZGENRE.MUSIC.ALTERNATIVE=',  'CONTEMPORARY ERA',  'WRAP',  'SINGING AS A SINGER',  'MOTOWN',  'RELIGIOUS MUSIC',  'CCM',  'ROCK & ROLL',  'LINING',  'FOLK-ROCK',  'CONTEMPORARY COUNTRY MUSIC',  'HARD BOP',  'ORIGINAL BANDS',  'SINGER-SONGWRITERS',  'UK HIP-HOP',  'DABSTEP',  'CONTEMPORARY SINGER-SONGWRITER',  'SPOKEN WORD',  'SERTANELLO',  'CONTEMPORARY SINGER/SONGWRITER',  'CLASSICAL CROSSOVER',  'BOLLYWOOD',  'AMERICAN TRAD ROCK',  'URBAN LATINO-AMERICANA',  'CUTTING EDGE JAZZ',  'DISCO',  'KOREAN INDIE MUSIC']

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

