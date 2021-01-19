import os
import numpy as np
import glob
import librosa
import fire
import tqdm
import multiprocessing
from multiprocessing import Pool

class Processor:
        def __init__(self):
        	self.fs = 16000

        def create_paths(self):
        	self.npy_path = "/home/jupyter/models/jamendo-mood/dataset-audios-npy/"
        	if not os.path.exists(self.npy_path):
        		os.makedirs(self.npy_path)

        def get_npy(self, fn):
        	x, sr = librosa.core.load(fn, sr=self.fs)
        	return x

        def iterate(self, data_path):
                self.create_paths()
                self.data_path = data_path
                self.files = list(glob.iglob(data_path + '**/*.mp3', recursive=True)) + list(glob.iglob(data_path + '**/*.m4a', recursive=True))
                p = Pool(multiprocessing.cpu_count())
                list(tqdm.tqdm(p.imap(self.process_file,self.files),total=len(self.files)))

        def process_file(self,fn):
                        fn = os.path.join(self.data_path,fn)
                        npy_fn = os.path.join(self.npy_path, fn.split('/')[-1][:-3]+'npy')
                        if not os.path.exists(npy_fn):
                                try:
                                        x = self.get_npy(fn)
                                        np.save(open(npy_fn, 'wb'), x)
                                except RuntimeError:
                                        # some audio files are broken
                                        print(fn)

if __name__ == '__main__':
	p = Processor()
	fire.Fire({'run': p.iterate})
