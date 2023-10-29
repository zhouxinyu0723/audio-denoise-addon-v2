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
from ZENNet.signalprocessing.sftf import hann_window, isftf_recover_coef, sftf_a_trace, isftf_a_trace, sftf_a_frame

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
assert target_samples//window_len >= 30

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
		# noise_F has shape (frames, N, window_size, channels)
		if(model.is_recursive):
			# iterate throught frames
			hidden_state = None
			recover_F = torch.zeros(noisy_F.shape, dtype=torch.cfloat).to(device)
			for i in range(noisy_F.shape[0]):
				hidden_state, suppression_hint = model(hidden_state, torch.log(noisy_F[i,:,:,:].abs()))
				recover_F[i,:,:,:] = torch.polar(noisy_F[i,:,:,:].abs()*suppression_hint, noisy_F[i,:,:,:].angle())
			recover = isftf_a_trace(recover_F, window_stride, sftf_window, isftf_window)

			
			# def log_mse_loss(recover, voice):
			# 	avg_db = torch.mean(torch.log(voice+1e-7), 1, True)
			# 	min_db = avg_db-40
			# 	loss = mseloss(torch.clamp(torch.log(recover+1e-7), min=min_db), torch.clamp(torch.log(voice+1e-7), min=min_db))
			# 	return loss
			# if epoch < 30:
			# 	voice_F = sftf_a_trace(voice, window_stride, sftf_window)
			# 	loss = mseloss(recover_F.abs(), voice_F.abs())
			# 	loss_ref = mseloss(noisy_F.abs(), voice_F.abs())
			# 	print(loss/loss_ref*100)
			# else:
			# 	voice_F = sftf_a_trace(voice, window_stride, sftf_window)
			# 	loss = mseloss(recover_F[target_samples//window_len//2:,:,:,:].abs(), voice_F[target_samples//window_len//2:,:,:,:].abs())
			# 	loss_ref = mseloss(noisy_F[target_samples//window_len//2:,:,:,:].abs(), voice_F[target_samples//window_len//2:,:,:,:].abs())
			# 	print(loss/loss_ref*100)

			# # loss
			# def log_mse_loss(recover, voice):
			# 	avg_db = torch.mean(torch.log(voice+1e-7), 1, True)
			# 	min_db = avg_db-40
			# 	loss = mseloss(torch.clamp(torch.log(recover+1e-7), min=min_db), torch.clamp(torch.log(voice+1e-7), min=min_db))
			# 	return loss
			
			# if epoch < 30:
			# 	shift = randint(-window_len, window_len)
			# 	large_window_len = randint(window_len, 4*window_len)
			# 	large_sftf_window = hann_window(large_window_len).to(device)[None,:,None]
			# 	large_stride = 4*large_window_len

			# 	full_trace_loss = log_mse_loss(sftf_a_trace(recover[:,2*window_len+shift:-2*window_len+shift,:], large_stride, large_sftf_window).abs(), sftf_a_trace(voice[:,2*window_len+shift:-2*window_len+shift,:], large_stride, large_sftf_window).abs())
			# 	full_trace_loss_ref = log_mse_loss(sftf_a_trace(noisy[:,2*window_len+shift:-2*window_len+shift,:], large_stride, large_sftf_window).abs(), sftf_a_trace(voice[:,2*window_len+shift:-2*window_len+shift,:], large_stride, large_sftf_window).abs())
			# 	print(full_trace_loss/full_trace_loss_ref*100)

			# trail_start = randint(target_samples/2, target_samples-5*window_len)
			# trail_end = randint(trail_start+window_len, trail_start+4*window_len)
			# trace_tail_loss = log_mse_loss(sftf_a_frame(recover[:,trail_start:trail_end,:]).abs(), sftf_a_frame(voice[:,trail_start:trail_end,:]).abs())
			# trace_tail_loss_ref = log_mse_loss(sftf_a_frame(noisy[:,trail_start:trail_end,:]).abs(), sftf_a_frame(voice[:,trail_start:trail_end,:]).abs())
			# print(trace_tail_loss/trace_tail_loss_ref*100)
			# fade_coeff = 0.5/(epoch/10+1)
			# if epoch < 30:
			# 	loss = fade_coeff*full_trace_loss + (1-fade_coeff)*trace_tail_loss
			# else:
			# 	loss = trace_tail_loss
			# loss finish

			# loss
			def sdr_loss(recover, voice):
				return -torch.sum(recover*voice)/torch.sqrt(torch.sum(recover**2)*torch.sum(voice**2))
			
			if epoch < 30:
				full_trace_loss = sdr_loss(recover[:,window_len:-window_len,:], voice[:,window_len:-window_len,:])#+ sdr_loss((noisy - recover)[:,window_len:-window_len,:], noise[:,window_len:-window_len,:])
				full_trace_loss_ref = sdr_loss(noisy[:,window_len:-window_len,:], voice[:,window_len:-window_len,:])
				print(full_trace_loss)
				print(full_trace_loss/full_trace_loss_ref*100)

			trail_start = target_samples//2
			trail_end = target_samples-window_len
			trace_tail_loss = sdr_loss(recover[:,trail_start:trail_end,:], voice[:,trail_start:trail_end,:])#+ sdr_loss((noisy - recover)[:,trail_start:trail_end,:], noise[:,trail_start:trail_end,:])
			trace_tail_loss_ref = sdr_loss(noisy[:,trail_start:trail_end,:], voice[:,trail_start:trail_end,:])
			print(trace_tail_loss)
			print(trace_tail_loss/trace_tail_loss_ref*100)

			fade_coeff = 0.5/(epoch/10+1)
			if epoch < 30:
				loss = fade_coeff*full_trace_loss + (1-fade_coeff)*trace_tail_loss
			else:
				loss = trace_tail_loss

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
					hidden_state = None
					recover_F = torch.zeros(noisy_F.shape, dtype=torch.cfloat).to(device)
					for i in range(noisy_F.shape[0]):
						hidden_state, suppression_hint = model(hidden_state, noisy_F[i,:,:,:].abs())
						recover_F[i,:,:,:] = torch.polar(noisy_F[i,:,:,:].abs()*suppression_hint, noisy_F[i,:,:,:].angle())
					recover = isftf_a_trace(recover_F, window_stride, sftf_window, isftf_window)
				else:
					raise NotImplementedError
				torchaudio.save(dump_dir/f"origin_{epoch}.wav", voice[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"noisy_{epoch}.wav", noisy[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"recover_{epoch}.wav", recover[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
			
