from math import pi
from random import randint
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchaudio
from ZENNet.dataset.dataset import ZENNetAudioDataset
from ZENNet.loss.loss import mseloss
from ZENNet.model.simple import Simple
from ZENNet.model.simle_fft import Simple_fft
from ZENNet.model.simple_norecursive import Simple_NR

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
chunk_overlap = 3
chunk_size =4410 *40*2//1000 //chunk_overlap*chunk_overlap	# target at 40ms
chunk_stride = chunk_size // chunk_overlap 
angle_speed = pi/chunk_size
#windowF2T = torch.hann_window(chunk_size)
windowF2T = torch.square(torch.sin((torch.arange(chunk_size)+0.5)*angle_speed))
windowT2F = windowF2T
_windowDIV = torch.zeros(windowF2T.shape)
_windowDIV_template = windowF2T*windowT2F
for i in range(chunk_overlap):
	if i==0:
		_windowDIV += _windowDIV_template.clone()
	else:
		_windowDIV[i*chunk_stride:] += _windowDIV_template[:chunk_size-i*chunk_stride]
		_windowDIV[:chunk_size-i*chunk_stride] += _windowDIV_template[i*chunk_stride:]
_windowMUL = 1/_windowDIV

#for i in range((target_samples-chunk_size)//chunk_stride +1):
#	start = i*chunk_stride
#	end = start+chunk_size
#	windowDIV[start: end] += _windowDIV_template
#windowMUL = 1/windowDIV
windowMUL = torch.zeros(target_samples)
for i in range(target_samples//chunk_size):
 start = i*chunk_size
 end = start+chunk_size
 windowMUL[start: end] += _windowMUL


#fig, ax = plt.subplots(3,1)
#ax[0].plot(np.linspace(0, 100, chunk_size), _windowDIV_template.numpy())
#ax[1].plot(np.linspace(0, 100, chunk_size), _windowMUL.numpy())
#ax[2].plot(np.linspace(0, 100, target_samples), windowMUL.numpy())
#plt.show()

windowF2T = windowF2T.to(device)[None,:,None]
windowT2F = windowT2F.to(device)[None,:,None]
_windowMUL = _windowMUL.to(device)[None,:,None]
windowMUL = windowMUL.to(device)[None,:,None]

#windowT2F = torch.sin((torch.arange(chunk_size)+0.5)*angle_speed)[None,:,None].to(device)



def recursive_loop(voice, noisy, model, loss_func):
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
	"""
	hidden_state = None
	batch_size = voice.shape[0]
	recover_all = torch.zeros([batch_size,target_samples, target_channels]).to(device)
	for i in range((target_samples-chunk_size)//chunk_stride +1):
		start = i*chunk_stride
		end = start+chunk_size
		with torch.no_grad():
			noisy_T = noisy[:,start:end,:]*windowT2F
			noisy_F = torch.fft.fft(noisy_T, dim = 1)
		hidden_state, suppression_hint = model(hidden_state, noisy_F.abs())
		recover_abs = noisy_F.abs()*suppression_hint
		recover_angle = noisy_F.angle()
		recover_F = torch.polar(recover_abs, recover_angle)
		recover_T = torch.fft.ifft(recover_F, dim = 1)
		recover_all[:,start: end,:] += recover_T.real*windowF2T
	recover_all *= windowMUL
	#print(model.scale.detach())
	#print(loss_func(voice[:,start:end,:], recover_all[:,start:end,:])*100)
	#print(loss_func(noisy[:,start:end,:], voice[:,start:end,:])*100)
	#return loss_func(voice[:,start:end,:], recover_all[:,start:end,:])
	random_shift = randint(chunk_stride*(chunk_overlap-1), chunk_size+chunk_stride*(chunk_overlap-1))
	end = max(chunk_size, target_samples - random_shift)
	start = end - chunk_size*4
	angle_speed = pi/chunk_size/4
	windowT2F_large = torch.square(torch.sin((torch.arange(chunk_size*4)+0.5)*angle_speed)).to(device)[None,:,None]
	recover_F_loss = torch.fft.fft(recover_all[:,start:end,:]*windowT2F_large, dim = 1)
	recover_F_loss_abs = recover_F_loss.abs()
	recover_F_loss_angle = recover_F_loss.angle()
	with torch.no_grad():
		voice_F_loss = torch.fft.fft(voice[:,start:end,:]*windowT2F_large, dim = 1)
		voice_F_loss_abs = voice_F_loss.abs()
		voice_F_loss_angle = voice_F_loss.angle()
	loss = 0.7*loss_func(recover_F_loss_abs, voice_F_loss_abs) 
	loss += 0.3*loss_func(recover_F_loss_angle, voice_F_loss_angle) 
	print(loss*100)
	print(loss_func(noisy[:,start:end,:], voice[:,start:end,:])*100)
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
	model = Simple_fft(target_samples, chunk_size, target_channels, device)
	model.to(device)

	# create loss function
	loss_func = mseloss

	# create the optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
	#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 )

	running_loss = 0.0
	for epoch in range(3000):
		model.train()
		optimizer.zero_grad()
		voice = next(iter(chinese_dataloader))
		noise = next(iter(noise_dataloader))
		# voice, noise has same shape (N, samples, channels)
		noisy = voice + noise

		hidden_state = None
		if(model.is_recursive):
			loss, recover_all = recursive_loop(voice, noisy, model, loss_func)
		else:
			recover = model(noisy)
			loss = loss_func(recover, voice)
		#loss.backward()
		#torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
		#optimizer.step()
		#weight =model.linear.weight.detach().clone()
		print(f"epoch {epoch}")
		#print(torch.sum(weight,1))
		#print(mseloss(voice[:,start:end,:]+noise[:,start:end,:], voice[:,start: end,:]))

		if(epoch%10==1):
			model.eval()
			with torch.no_grad():
				if(model.is_recursive):
					loss, recover_all = recursive_loop(voice, noisy, model, mseloss)
				else:
					recover_all = model(noisy)
				torchaudio.save(dump_dir/f"origin_{epoch}.wav", voice[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"noisy_{epoch}.wav", noise[0,:,:].to("cpu")
+	voice[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"recover_{epoch}.wav", recover_all[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
			
