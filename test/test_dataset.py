from pathlib import Path
from ZENNet.dataset.dataset import ZENNetAudioDataset

def test_load_dataset():
	clean__audio_dataset = ZENNetAudioDataset([Path("/home/zhou/Data/study/denoiser/dnn/ZENNet/dataset/other_dataset/tut/400515/TUT-acoustic-scenes-2017-development.audio.1/TUT-acoustic-scenes-2017-development")])

	sound = clean__audio_dataset[22]
	sound = clean__audio_dataset[clean__audio_dataset.__len__()-1]
