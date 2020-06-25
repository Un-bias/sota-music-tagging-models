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

TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']

def get_model(model_type):
    if model_type == 'fcn':
        input_length = 29 * 16000
        model = Model.FCN()
    elif model_type == 'musicnn':
        input_length = 3 * 16000
        model = Model.Musicnn(dataset=self.dataset)
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


if __name__ == '__main__':
    batch_size=16
    dataset='jamendo' # choices=['mtat', 'msd', 'jamendo']
    model_type='fcn' # choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'boc', 'boc_res', 'attention', 'hcnn']
    model_load_path='../models/jamendo/fcn/best_model.pth'
    input_file="/content/temp.m4a"

    #load_remote_file(handle, audio_file_path)
    normalize(input_file,input_file.replace(".m4a","mp3"))
    prediction = predict(model_type,input_file.replace(".m4a","mp3"), batch_size, model_load_path)

    jamendo_df = pd.DataFrame([list(prediction)],columns=TAGS)
    print(jamendo_df)
