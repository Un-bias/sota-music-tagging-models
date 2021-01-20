# coding: utf-8
import pickle
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer

META_PATH = '/home/jupyter/models/sota/split/genres'

TAGS = ['R&B/SOUL', 'INDIE POP', 'ALTERNATIVE FOLK', 'LATIN URBAN', 'ALTERNATIVE', 'DANCE', 'JAZZ', 'POP', 'LATINO', 'HIP-HOP/RAP', 'METAL', 'ROCK', 'AUTHOR-PERFORMER', 'ELECTRONIC', 'ROCK INDIE', 'HOUSE', 'CHRISTIAN & GOSPEL', 'INDEPENDENT ROCK', 'VOCAL', 'SERTANEJO', 'LATIN POP', 'DOWNTEMPO', 'SINGER/SONGWRITER', 'NEO-SOUL', 'BRAZILIAN', 'SPANISH POP', 'K-POP', 'LATIN JAZZ', 'TECHNO', 'INTERNATIONAL POP', 'COUNTRY', 'HIP HOP / RAP', 'BLUES', 'POP IN SPANISH', 'CLASSICAL', 'SONG WRITERS', 'INDIE-ROCK', 'SINGER-SONGWRITER', 'R & B / SOUL', 'FOLK', 'IDM/EXPERIMENTAL', 'INDIE ROCK', 'SOUL', 'CHRISTIAN AND GOSPEL', 'CONTEMPORARY JAZZ', 'SOUNDTRACKS', 'SOUNDTRACK', 'DANCE MUSIC', 'LATIN AMERICAN', 'FUSION', 'LOUNGE', 'MEXICAN MUSIC', 'NEW WAVE', 'PAGODA', 'AMBIENT', 'ELECTONIC MUSIC', "JUNGLE/DRUM'N'BASS", 'GOSPEL', 'RELIGIOUS MUSIC', 'HIPHOP/RAP', 'PERFORMER / COMPOSER', 'ALTERNATIVE FOLK MUSIC', 'LATIN', 'ELECTRONIC MUSIC', 'BASS', 'FOLK ALTERNATIVO', 'SALSA AND TROPICS', 'URBAN LATIN', 'URBAN LATINO-AMERICANA', 'TRANCE', 'AVANT-GARDE JAZZ', 'TROPICAL MUSIC', 'INSTRUMENTAL', 'CHRISTMAS', 'WORLD', 'PUNK', 'FUNK', 'FITNESS & WORKOUT', 'FUNK PARTY', 'WRAP', 'WORLD MUSIC', 'DUBSTEP', 'CHRISTIAN MUSIC', 'PSYCHEDELIC', 'ALTERNATIVE RAP', 'JAZZ CROSSOVER', 'R&B/SAO LINGLE', 'CONTEMPORARY FOLK', 'REGGAE', 'SALSA AND TROPICAL', 'VOCAL MUSIC', 'INDEPENDENT POP', 'GARAGE', 'AFROBEATS', 'INDIE-POP', 'FOLKROCK', 'LATIN MUSIC', 'LINING', 'OLD SCHOOL RAP', 'FRENCH POP', 'VOCAL JAZZ', 'BIG BAND', 'ALTERNATIVE - PEOPLE', 'WORLDWIDE', 'MAINSTREAM JAZZ', 'WEST COAST RAP', 'J-POP', 'POP/ROCK', 'CONTEMPORARY COUNTRY', 'MODERN ERA', 'JAPAN', 'ORIGINAL BANDS', 'HARD ROCK', 'DISCO', 'URBAN LATIN MUSIC', 'ALTERNATIVE AND ROCK IN SPANISH', 'MUSIC', 'TRADITIONAL FOLK MUSIC', 'MOTOWN', 'ROCK: INDEPENDENT', 'AFRO-POP', 'IDM / EXPERIMENTAL', 'CONTEMPORARY R&B', 'BREAKBEAT', 'LATIN RAP', 'POP INDIE', 'FOLK-ROCK', 'ADULT CONTEMPORARY', 'DABSTEP', 'CLASSIC', 'HIP HOP', 'LOCK', 'GOTH ROCK', 'ADULT ALTERNATIVE', 'HOLIDAY', 'LATINA', 'DIRTY SOUTH', 'AMERICANA', 'EASY LISTENING', 'ROOTS ROCK', 'SMOOTH JAZZ', 'LATIN AMERICAN URBAN MUSIC', 'SINGER-SONGWRITERS', 'SPOKEN WORD', 'DUB', 'SERTANELLO', 'CONTEMPORARY PERFORMER / COMPOSER', 'OLDIES', 'CONTEMPORARY ERA', 'HIPHOP', 'POP MUSIC IN SPANISH', 'MODERN DANCEHALL', 'BRITPOP', 'NEOSOUL', 'NEO SOUL', "CHILDREN'S MUSIC", 'POP PUNK', 'LATIN METROPOLITAN MUSIC', 'INDIAN POP', 'BEL CANTO', 'SOUND BAND', 'BLUES-ROCK', 'CONTEMPORARY SINGER/SONGWRITER', 'SEOUL', '=MZGENRE.MUSIC.ALTERNATIVE=', 'ORIGINAL SOUND TRACK', 'FESTIVAL', 'ROCK & ROLL', 'IT WAS MODERN', 'TRADITIONAL FOLK', 'ORIGINAL SCORE', 'JAZZ VOCAL', 'CCM', 'AMERICAN TRADITIONAL ROCK', 'R&B CONTEMPORAIN', 'ELECTRONICALLY', 'CUTTING EDGE JAZZ', 'GRUNGE', 'HARD BOP', 'KOREAN INDIE MUSIC', 'COLLEGE ROCK', 'BOLLYWOOD', 'SINGING AS A SINGER', 'INDUSTRIAL', 'LENSES', 'UK HIP-HOP', 'NEW AGE', 'SLOW SHOT', 'INSPIRATIONAL', '=DANCE=', 'AMERICAN TRAD ROCK', 'CLASSICAL CROSSOVER', 'LATIN WRAP', 'BRAZILIAN MUSIC', 'CONTEMPORARY COUNTRY MUSIC', 'SPANISH POP MUSIC', 'TRADITIONAL AMERICAN ROCK', 'NOW SOUL', 'EAST COAST RAP', 'UNDERGROUND RAP', 'TURKISH', 'AFRO HOUSE', 'CONTRIBUTING AUTHORS', 'SURF', 'CONTEMPORARY SINGER-SONGWRITER', 'ALTERNATIVE - RAP']

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

