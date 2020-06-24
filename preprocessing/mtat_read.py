import os
import numpy as np
import glob
import librosa
import fire
import tqdm


class Processor:
	def __init__(self):
		self.fs = 16000

	def create_paths(self, data_path):
		self.npy_path = os.path.join(data_path, 'mtat', 'npy')
		if not os.path.exists(self.npy_path):
			os.makedirs(self.npy_path)

	def get_npy(self, fn):
		x, sr = librosa.core.load(fn, sr=self.fs)
		return x

	def iterate(self, data_path):
		self.create_paths(data_path)
		self.files = os.listdir(os.path.join(data_path,"mp3"))
		for fn in tqdm.tqdm(self.files):
			npy_fn = os.path.join(self.npy_path, fn.split('/')[-1][:-3]+'npy')
			if not os.path.exists(npy_fn):
				try:
					x = self.get_npy(fn)
					np.save(open(npy_fn, 'wb'), x)
				except RuntimeError:
					# some audio files are broken
					print(fn)
					continue

if __name__ == '__main__':
	p = Processor()
	fire.Fire({'run': p.iterate})
