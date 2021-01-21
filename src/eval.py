# coding: utf-8
import librosa
from pydub import AudioSegment
import torch
import torch.nn as nn
from torch.autograd import Variable
import model as Model
import numpy as np
import pandas as pd
import os

jamendomood_tags = ["mood/theme---action", "mood/theme---adventure", "mood/theme---advertising", "mood/theme---background", "mood/theme---ballad", "mood/theme---calm", "mood/theme---children", "mood/theme---christmas", "mood/theme---commercial", "mood/theme---cool", "mood/theme---corporate", "mood/theme---dark", "mood/theme---deep", "mood/theme---documentary", "mood/theme---drama", "mood/theme---dramatic", "mood/theme---dream", "mood/theme---emotional", "mood/theme---energetic", "mood/theme---epic", "mood/theme---fast", "mood/theme---film", "mood/theme---fun", "mood/theme---funny", "mood/theme---game", "mood/theme---groovy", "mood/theme---happy", "mood/theme---heavy", "mood/theme---holiday", "mood/theme---hopeful", "mood/theme---inspiring", "mood/theme---love", "mood/theme---meditative", "mood/theme---melancholic", "mood/theme---melodic", "mood/theme---motivational", "mood/theme---movie", "mood/theme---nature", "mood/theme---party", "mood/theme---positive", "mood/theme---powerful", "mood/theme---relaxing", "mood/theme---retro", "mood/theme---romantic", "mood/theme---sad", "mood/theme---sexy", "mood/theme---slow", "mood/theme---soft", "mood/theme---soundscape", "mood/theme---space", "mood/theme---sport", "mood/theme---summer", "mood/theme---trailer", "mood/theme---travel", "mood/theme---upbeat", "mood/theme---uplifting"]
jamendo_tags = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']
msd_tags = ["rock", "pop", "alternative", "indie", "electronic", "female vocalists", "dance", "00s", "alternative rock", "jazz", "beautiful", "metal", "chillout", "male vocalists", "classic rock", "soul", "indie rock", "Mellow", "electronica", "80s", "folk", "90s", "chill", "instrumental", "punk", "oldies", "blues", "hard rock", "ambient", "acoustic", "experimental", "female vocalist", "guitar", "Hip-Hop", "70s", "party", "country", "easy listening", "sexy", "catchy", "funk", "electro", "heavy metal", "Progressive rock", "60s", "rnb", "indie pop", "sad", "House", "happy"]
mtat_tags = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']
genres_tags = ['ALTERNATIVE',  'K-POP',  'SPANISH POP',  'HIP-HOP/RAP',  'R&B/SOUL',  'DANCE',  'POP',  'LATIN URBAN',  'LATIN POP',  'INDIE POP',  'INDIE ROCK',  'ELECTRONIC',  'JAZZ',  'ROCK',  'COUNTRY',  'MEXICAN MUSIC',  'FUSION',  'FUNK',  'ALTERNATIVE RAP',  'SINGER/SONGWRITER',  'BRAZILIAN',  'ALTERNATIVE FOLK',  'SOUNDTRACK',  'LATINO',  'PUNK',  'TRANCE',  'FRENCH POP',  'ELECTONIC MUSIC',  'LATIN MUSIC',  'SOUNDTRACKS',  'EASY LISTENING',  'HOUSE',  'CONTEMPORARY R&B',  'TECHNO',  'METAL',  'JAZZ VOCAL',  'WORLD',  'CHRISTIAN & GOSPEL',  'SOUND BAND',  'BREAKBEAT',  'JAZZ CROSSOVER',  'SERTANEJO',  'HIP HOP / RAP',  'NEO-SOUL',  'CONTEMPORARY JAZZ',  'VOCAL',  'TRADITIONAL FOLK',  'SALSA AND TROPICAL',  'URBAN LATIN',  'DOWNTEMPO',  'INDIE-ROCK',  'LATIN',  'HARD ROCK',  'PERFORMER / COMPOSER',  'INSPIRATIONAL',  'REGGAE',  'BIG BAND',  'POP IN SPANISH',  'R & B / SOUL',  'ALTERNATIVE FOLK MUSIC',  'INDIE-POP',  'SINGER-SONGWRITER',  'POP INDIE',  'IDM/EXPERIMENTAL',  'MODERN ERA',  'INDEPENDENT ROCK',  'J-POP',  'WORLD MUSIC',  'FUNK PARTY',  'CHRISTMAS',  'HOLIDAY',  'INTERNATIONAL POP',  'CONTEMPORARY FOLK',  'NEW WAVE',  'AUTHOR-PERFORMER',  'HIPHOP/RAP',  'FITNESS & WORKOUT',  'HIPHOP',  'ELECTRONIC MUSIC',  'COLLEGE ROCK',  'INSTRUMENTAL',  'CHRISTIAN MUSIC',  'TROPICAL MUSIC',  'SEOUL',  'AVANT-GARDE JAZZ',  'R&B/SAO LINGLE',  "JUNGLE/DRUM'N'BASS",  'AMBIENT',  'FOLK',  "CHILDREN'S MUSIC",  'URBAN LATIN MUSIC',  'ORIGINAL SCORE',  'SMOOTH JAZZ',  'WORLDWIDE',  'VOCAL JAZZ',  'DANCE MUSIC',  'SOUL',  'NEOSOUL',  'GRUNGE',  'FESTIVAL',  'IT WAS MODERN',  'LATIN AMERICAN',  '=DANCE=',  'CLASSICAL',  'UNDERGROUND RAP',  'SONG WRITERS',  'PSYCHEDELIC',  'OLD SCHOOL RAP',  'BASS',  'ADULT ALTERNATIVE',  'OLDIES',  'GARAGE',  'LATIN AMERICAN URBAN MUSIC',  'MODERN DANCEHALL',  'LATINA',  'ROCK INDIE',  'DUB',  'ADULT CONTEMPORARY',  'TRADITIONAL FOLK MUSIC',  'INDIAN POP',  'SLOW SHOT',  'R&B CONTEMPORAIN',  'ROCK: INDEPENDENT',  'LATIN JAZZ',  'POP PUNK',  'DUBSTEP',  'LOCK',  'BRAZILIAN MUSIC',  'NEO SOUL',  'ELECTRONICALLY',  'CONTRIBUTING AUTHORS',  'CONTEMPORARY COUNTRY',  'NEW AGE',  'LOUNGE',  'ORIGINAL SOUND TRACK',  'AFRO-POP',  'AMERICANA',  'HIP HOP',  'INDUSTRIAL',  'GOTH ROCK',  'BLUES',  'LATIN RAP',  'GOSPEL',  'MUSIC',  'BLUES-ROCK',  'IDM / EXPERIMENTAL',  'POP/ROCK',  'VOCAL MUSIC',  '=MZGENRE.MUSIC.ALTERNATIVE=',  'CONTEMPORARY ERA',  'WRAP',  'SINGING AS A SINGER',  'MOTOWN',  'RELIGIOUS MUSIC',  'CCM',  'ROCK & ROLL',  'LINING',  'FOLK-ROCK',  'CONTEMPORARY COUNTRY MUSIC',  'HARD BOP',  'ORIGINAL BANDS',  'SINGER-SONGWRITERS',  'UK HIP-HOP',  'DABSTEP',  'CONTEMPORARY SINGER-SONGWRITER',  'SPOKEN WORD',  'SERTANELLO',  'CONTEMPORARY SINGER/SONGWRITER',  'CLASSICAL CROSSOVER',  'BOLLYWOOD',  'AMERICAN TRAD ROCK',  'URBAN LATINO-AMERICANA',  'CUTTING EDGE JAZZ',  'DISCO',  'KOREAN INDIE MUSIC']

def tags(dataset):
    if dataset == "jamendo":
        return jamendo_tags
    elif dataset == "jamendo-mood":
        return jamendomood_tags
    elif dataset == "msd":
        return msd_tags
    elif dataset == "mtat":
        return mtat_tags
    elif dataset == "genres":
        return genres_tags
    else:
        raise Exception("Invalid dataset")

def get_model(model_type):
    if model_type == 'fcn':
        input_length = 29 * 16000
        model = Model.FCN()
    elif model_type == 'musicnn':
        input_length = 3 * 16000
        model = Model.Musicnn()
    elif model_type == 'crnn':
        input_length = 29 * 16000
        model = Model.CRNN()
    elif model_type == 'sample':
        input_length = 59049
        model = Model.SampleCNN()
    elif model_type == 'se':
        input_length = 59049
        model = Model.SampleCNNSE()
    elif model_type == 'boc':
        input_length = 59049
        model = Model.BoCCNN()
    elif model_type == 'boc_res':
        input_length = 59049
        model = Model.BoCCNN_Res()
    elif model_type == 'attention':
        input_length = 15 * 16000
        model = Model.CNNSA()
    elif model_type == 'hcnn':
        input_length = 5 * 16000
        model = Model.HarmonicCNN()
    else:
        raise Exception('model_type has to be one of [fcn, musicnn, crnn, sample, se, boc, boc_res, attention]')
    return model, input_length

def load(model, filename):
    S = torch.load(filename)
    if 'spec.mel_scale.fb' in S.keys():
        model.spec.mel_scale.fb = S['spec.mel_scale.fb']
    model.load_state_dict(S)

def build_model(model_type,model_load_path):
    model, input_length = get_model(model_type)
    # cuda
    model.cuda()
    # load model
    load(model, model_load_path)
    return model, input_length

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

fs = 16000
def normalize(fn,output_file):
    audio = AudioSegment.from_file(fn)
    audio = audio.set_frame_rate(fs)
    audio.export(output_file, format="mp3")

def get_npy(fn):
    x, sr = librosa.core.load(fn, sr=fs)
    return x

def get_tensor(fn, input_length, batch_size):
    raw = get_npy(fn)
    length = len(raw)
    hop = (length - input_length) // batch_size
    x = torch.zeros(batch_size, input_length)
    for i in range(batch_size):
        x[i] = torch.Tensor(raw[i*hop:i*hop+input_length]).unsqueeze(0)
    return x

def predict(model_type,fn, batch_size, model_load_path):
    model, input_length = build_model(model_type,model_load_path)
    model = model.eval()
    est_array = []
    gt_array = []
    losses = []
    reconst_loss = nn.BCELoss()

    x = get_tensor(fn, input_length, batch_size)

    # forward
    x = to_var(x)
    out = model(x)
    out = out.detach().cpu()

    # estimate
    prediction = np.array(out).mean(axis=0)
    return prediction

def predict_time_series(fn, batch_size, model_load_path):
    model, input_length = build_model('musicnn',model_load_path)
    model = model.eval()
    est_array = []
    gt_array = []
    losses = []
    reconst_loss = nn.BCELoss()

    x = get_tensor(fn, input_length, batch_size)

    # forward
    x = to_var(x)
    out = model(x)
    out = out.detach().cpu()

    # estimate
    prediction = np.array(out)
    return prediction

def run_time_series(input_file, dataset='mtat',batch_size=16):
  model_type='musicnn'
  model_load_path=f'../models/{dataset}/{model_type}/best_model.pth'

  #load_remote_file(handle, audio_file_path)
  normalize(input_file,"temp.mp3")
  prediction = predict_time_series("temp.mp3", batch_size, model_load_path).tolist()
  
  return pd.DataFrame(prediction,columns=tags(dataset))


def run(input_file, dataset='mtat', model_type='fcn',batch_size=16):
  # dataset choices=['mtat', 'msd', 'jamendo', 'jamendo-mood', 'genres']
  # model_type choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'boc', 'boc_res', 'attention', 'hcnn']
  model_load_path=f'../models/{dataset}/{model_type}/best_model.pth'

  #load_remote_file(handle, audio_file_path)
  normalize(input_file,"temp.mp3")
  prediction = predict(model_type,"temp.mp3", batch_size, model_load_path)
  
  return pd.DataFrame([list(prediction)],columns=tags(dataset))

if __name__ == '__main__':
    output = run("data/mp3/reggae.mp3")
    print(output)
