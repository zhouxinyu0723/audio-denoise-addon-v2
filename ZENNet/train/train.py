from math import pi
from random import randrange, randint
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchaudio
from ZENNet.dataset.dataset import ZENNetAudioDataset
from ZENNet.loss.loss import mseloss, sftf_loss_prepare
from ZENNet.model.simple import Simple
from ZENNet.model.simle_fft import Simple_fft
from ZENNet.model.simleStftGru import SimpleStftGru
from ZENNet.model.simple_norecursive import Simple_NR
from ZENNet.signalprocessing.sftf import hann_window, isftf_recover_coef, sftf_a_trace, isftf_a_trace

# config device
if torch.cuda.is_available():
	device = "cuda"
	import gc
	gc.collect()
	torch.cuda.empty_cache()
else:
	device = "cpu"

# some arguments
dump_dir = Path("/home/zhou/Data/study/denoiser/dnn/ZENNet/dump")
target_stride = 4000
target_sample_rate = 44100
target_samples =220500 
target_channels =2
window_len = 4410 *40*2//1000 # target at 40ms
chunk_overlap = 3

# round up arguments
# so that target_samples are multiples of window_len
# so that window_len are multiples of window_stride
window_len = window_len//chunk_overlap*chunk_overlap	
window_stride = window_len // chunk_overlap
target_samples = target_samples//window_len*window_len
assert target_samples >= 2*window_len

# prepare windows
def isftf_recover_coeff_all(sftf_window, isftf_window, chunk_stride, target_samples):
	chunk_size = sftf_window.shape[0]
	recover_coeff = isftf_recover_coef(sftf_window, isftf_window, chunk_stride)
	recover_coeff_all = torch.zeros(target_samples).to(device)
	for i in range(target_samples//chunk_size):
		start = i*chunk_size
		end = start+chunk_size
		recover_coeff_all[start: end] = recover_coeff
	return recover_coeff_all

isftf_window = hann_window(window_len).to(device)
sftf_window = hann_window(window_len).to(device)
recover_coeff_all = isftf_recover_coeff_all(sftf_window, isftf_window, window_stride, target_samples)
isftf_window = isftf_window[None, :, None]
sftf_window = sftf_window[None, :, None]
recover_coeff_all = recover_coeff_all[None, :, None]

def recursive_loop(voice, noisy, model):
	"""
	Brief
	----------
	This also contains part of the model
	It contains STFT, ISTFT, suppression(multiplication) and loss function
	The model instance will take STFT result and give suppression hint

	Parameters
	----------
	voice : Tensor
		Input tensor of shape (batch_size, target_samples, channels)
	noisy : Tensor
		Input tensor of shape (batch_size, target_samples, channels)
	model : torch.nn.Module
		Input a recursive model, for example RNN

	Returns
	-------
	Tensor Value
		loss
	Tensor
		Output tensor of shape (batch_size, target_samples, channels)
		The recoverd audio
	"""
	hidden_state = None
	batch_size = voice.shape[0]
	recover_all = torch.zeros([batch_size,target_samples, target_channels]).to(device)
	avg_hidden = 0
	avg_hidden_std = 0
	for i in range((target_samples-window_len)//window_stride +1):
		start = i*window_stride
		end = start+window_len
		with torch.no_grad():
			noisy_T = noisy[:,start:end,:]*sftf_window
			noisy_F = torch.fft.fft(noisy_T, dim = 1)
		hidden_state, suppression_hint = model(hidden_state, noisy_F.abs())
		recover_abs = noisy_F.abs()*suppression_hint
		recover_angle = noisy_F.angle()
		recover_F = torch.polar(recover_abs, recover_angle)
		recover_T = torch.fft.ifft(recover_F, dim = 1)
		recover_all[:,start: end,:] += recover_T.real*isftf_window
		avg_hidden += torch.mean(hidden_state).detach().to("cpu").numpy().item()
		avg_hidden_std += torch.std(hidden_state).detach().to("cpu").numpy().item()
	avg_hidden /= (target_samples-window_len)//window_stride +1
	print(avg_hidden)
	print(avg_hidden_std)
	recover_all *= recover_coeff_all
	# TODO wrap the following into a new loss function
	#print(model.scale.detach())
	#print(loss_func(voice[:,start:end,:], recover_all[:,start:end,:])*100)
	#print(loss_func(noisy[:,start:end,:], voice[:,start:end,:])*100)
	#return loss_func(voice[:,start:end,:], recover_all[:,start:end,:])

	loss_window_size = window_len*randint(1, min(5, target_samples//window_len))
	start = randint(max(0,target_samples-loss_window_size-window_stride-2*window_len), target_samples-loss_window_size-window_len)
	end = start + loss_window_size

	recover_F_loss_abs, recover_F_loss_angle, voice_F_loss_abs, voice_F_loss_angle = sftf_loss_prepare(recover_all[:,start:end,:], voice[:,start:end,:])
	avg_db = torch.mean(torch.log(voice_F_loss_abs+1e-7),  1, True)
	loss = mseloss(torch.clamp(torch.log(recover_F_loss_abs+1e-7), min=avg_db-20), torch.clamp(torch.log(voice_F_loss_abs+1e-7), min=avg_db-20))
	#loss += 0.3*loss_func(recover_F_loss_angle, voice_F_loss_angle) 
	print(torch.mean(suppression_hint).to("cpu"))
	print(loss.to("cpu")*100)

	noisy_F_loss_abs, noisy_F_loss_angle, voice_F_loss_abs, voice_F_loss_angle = sftf_loss_prepare(noisy[:,start:end,:], voice[:,start:end,:])
	loss_ref = mseloss(torch.clamp(torch.log(noisy_F_loss_abs+1e-7), min=avg_db-20), torch.clamp(torch.log(voice_F_loss_abs+1e-7), min=avg_db-20))
	print(loss_ref.to("cpu")*100)
	print(loss/loss_ref*100)

	return loss, recover_all

def train():
	print(f"use {device}")
	# prepare dataset
	#chunk_size = 4410
	chinese_dataset = ZENNetAudioDataset([Path("/home/zhou/Data/study/denoiser/dnn/DNS-Challenge/dataset/other_dataset/aishell")], target_stride, target_sample_rate, target_samples, target_channels, device)
	noise_dataset = ZENNetAudioDataset([Path("/home/zhou/Data/study/denoiser/dnn/ZENNet/dataset/other_dataset/tut")], target_stride, target_sample_rate, target_samples, target_channels, device)
	# prepare data loader
	chinese_dataloader = DataLoader(chinese_dataset, batch_size=8, shuffle=True)

	noise_dataloader = DataLoader(noise_dataset, batch_size=8, shuffle=True)

	# create the model
	model = SimpleStftGru(target_samples, window_len, target_channels, device)
	model.to(device)

	# create the optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 )

	running_loss = 0.0
	for epoch in range(3000):
		model.train()
		optimizer.zero_grad()
		voice = next(iter(chinese_dataloader))
		noise = next(iter(noise_dataloader))
		# voice, noise has same shape (N, samples, channels)
		noisy = voice + noise

		noisy_F = sftf_a_trace(noisy, window_stride, sftf_window)
		


		if(model.is_recursive):
			loss, recover_all = recursive_loop(voice, noisy, model)
		else:
			# recover = model(noisy)
			raise NotImplementedError
			# loss = loss_func(recover, voice)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
		optimizer.step()
		#weight =model.linear.weight.detach().clone()
		print(f"epoch {epoch}")
		#print(torch.sum(weight,1))
		#print(mseloss(voice[:,start:end,:]+noise[:,start:end,:], voice[:,start: end,:]))

		if(epoch%10==1):
			model.eval()
			with torch.no_grad():
				if(model.is_recursive):
					loss, recover_all = recursive_loop(voice, noisy, model)
				else:
					raise NotImplementedError
					# recover_all = model(noisy)
				torchaudio.save(dump_dir/f"origin_{epoch}.wav", voice[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"noisy_{epoch}.wav", noise[0,:,:].to("cpu")
+	voice[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"recover_{epoch}.wav", recover_all[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
			
