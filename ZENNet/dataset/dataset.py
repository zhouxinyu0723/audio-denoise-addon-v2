from torch.utils.data import Dataset, random_split, DataLoader
from torch import Tensor
import torch
from pathlib import Path
from typing import List
from torchaudio import load as audio_load
from torchaudio import info as audio_info
# https://github.com/pytorch/audio/blob/0c8c138c89960e6105d7b61677a9799ad9924904/torchaudio/backend/soundfile_backend.py#L221-L236
from speechbrain.dataio.preprocess import AudioNormalizer
from sounddevice import play as audio_play
import time
 
class AudioDirectory:
	def __init__(self, path: Path, concatable: False) -> None:
		self.path = path
		self.concatable = concatable

class ZENNetAudioDataset(Dataset):
	def __init__(self, directories: List[Path], target_stride = 4410, target_sample_rate = 44100, target_samples = 220500, target_channels=2, device="cpu"):
		"""Get an element with a default value.

		Parameters
		----------
		clean_dirs : List[Path]
		    Each path is a path of a directory under which there are audio files
		"""
		self.target_stride = target_stride
		self.target_sample_rate = target_sample_rate
		self.target_samples = target_samples
		self.target_channels = target_channels
		self.device = device
		self.sample_rate_convert = AudioNormalizer(target_sample_rate, mix='keep')  # this is used to convert sample rate
		# the main structure here are a lookup tables whose key is index and value is audio file path under directories and start offset
		self.lookup = {}
		self.index = 0
		for directory in directories:
			for path in directory.glob("**/*.wav"):
				#print(path)
				self.read_single_file_and_append_to_lookup(path)

	def read_single_file_and_append_to_lookup(self, path: Path):
		metadata = audio_info(path)
		origin_samples_in_file = int(metadata.num_frames)
		origin_sample_rate = int(metadata.sample_rate)
		converted_target_stride = self.target_stride * origin_sample_rate // self.target_sample_rate
		converted_target_samples = self.target_samples * origin_sample_rate // self.target_sample_rate + 100
		offset = 0
		while(offset + converted_target_samples < origin_samples_in_file):
			self.lookup[self.index] = (path, offset, converted_target_samples)
			offset += converted_target_stride
			self.index += 1

	def __len__(self) -> int:
		return len(self.lookup)

	def __getitem__(self, index) -> Tensor:
		"""
		Parameters
		----------
		index : int
		    Index of the audio file in the dataset
		Returns
		-------
		Tensor
		    The audio file in the dataset, dimesions are (target_samples/time, channels)
		"""
		path, offset, converted_target_samples = self.lookup[index]
		frame, sample_rate = audio_load(path, frame_offset = offset, num_frames = converted_target_samples, channels_first=False)
		frame = frame.to(self.device)
		# TODO sample rate conversion is very cpu intensive, considering writing a script to store them in file
		frame = self.sample_rate_convert(frame, sample_rate)[:self.target_samples,:] 
		if frame.shape[1]==1 and self.target_channels==2:
			frame=frame.repeat(1,2)
		if frame.shape[1]==2 and self.target_channels==1:
			frame=torch.mean(frame, 1, True)
		return frame


		
def main():
	clean__audio_dataset = ZENNetAudioDataset([Path("/home/zhou/Data/study/denoiser/dnn/ZENNet/dataset/other_dataset/tut")])#/400515/TUT-acoustic-scenes-2017-development.audio.1/TUT-acoustic-scenes-2017-development

	sound = clean__audio_dataset[22]
	print(sound.shape)
	print(sound[1000:2000,:])
	print(torch.max(sound))
	print(torch.mean(sound))
	audio_play(sound, samplerate=44100, blocking=True)

	clean__audio_dataset = ZENNetAudioDataset([Path("/home/zhou/Data/study/denoiser/dnn/DNS-Challenge/dataset/other_dataset/aishell")])

	sound = clean__audio_dataset[99]
	print(sound.shape)
	audio_play(sound, samplerate=44100, blocking=True)



if __name__ == "__main__":
	main()

