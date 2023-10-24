from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchaudio
from ZENNet.dataset.dataset import ZENNetAudioDataset
from ZENNet.loss.loss import mseloss
from ZENNet.model.simple import Simple
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
chunk_size =4410 *40*2//1000 //2*2	# target at 40ms
chunk_stride = chunk_size // 2 
window = torch.sqrt(torch.hann_window(chunk_size, device=device))[None,:,None]

def recursive_loop(voice, noisy, model, loss_func):
	"""
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
	for i in range((target_samples-chunk_size)//chunk_stride +1):
		start = i*chunk_stride
		end = start+chunk_size
		hidden_state, recover = model(hidden_state, noisy[:,start:end,:]*window)
	loss =loss_func(recover, voice[:,start:end,:]*window) 
	print(loss*100)
	print(loss_func(noisy[:,start:end,:]*window, voice[:,start:end,:]*window)*100)
	return loss

def recursive_loop_all(voice, noisy, model, loss_func):
	"""
	Parameters
	----------
	voice : Tensor
		Input tensor of shape (batch_size, target_samples, channels)
	nosie : Tensor
		Input tensor of shape (batch_size, target_samples, channels)
	model : torch.nn.Module
		Input a recursive model, for example RNN
	Returns
	-------
	Tensor
		Output tensor of shape (batch_size, target_samples, channels)
		the recoverd voice of last iteration
	"""
	# TODO add window funtion
	hidden_state = None
	batch_size = voice.shape[0]
	recover_all = torch.zeros([batch_size,target_samples, target_channels]).to(device)
	for iteration in range((target_samples-chunk_size)//chunk_stride +1):
		start = iteration*chunk_stride
		end = start+chunk_size
		hidden_state, recover = model(hidden_state, noisy[0,start:end,:]*window)
		recover_all[:,start: end,:] += recover.detach().clone()*window
	return None, recover_all

def train():
	print(f"use {device}")
	# prepare dataset
	#chunk_size = 4410
	chinese_dataset = ZENNetAudioDataset([Path("/home/zhou/Data/study/denoiser/dnn/DNS-Challenge/dataset/other_dataset/aishell")], target_stride, target_sample_rate, target_samples, target_channels, device)
	noise_dataset = ZENNetAudioDataset([Path("/home/zhou/Data/study/denoiser/dnn/ZENNet/dataset/other_dataset/tut")], target_stride, target_sample_rate, target_samples, target_channels, device)
	# prepare data loader
	chinese_dataloader = DataLoader(chinese_dataset, batch_size=64, shuffle=True)

	noise_dataloader = DataLoader(noise_dataset, batch_size=64, shuffle=True)

	# create the model
	model = Simple(target_samples, chunk_size, target_channels, device)
	model.to(device)

	# create loss function
	loss_func = mseloss

	# create the optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr=100)
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
			loss = recursive_loop(voice, noisy, model, loss_func)
		else:
			recover = model(noisy)
			loss = loss_func(recover, voice)
		loss.backward()
		#torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
		optimizer.step()
		weight =model.linear.weight.detach().clone()
		print(f"epoch {epoch}")
		#print(torch.sum(weight,1))
		#print(mseloss(voice[:,start:end,:]+noise[:,start:end,:], voice[:,start: end,:]))

		if(epoch%10==1):
			model.eval()
			with torch.no_grad():
				if(model.is_recursive):
					loss, recover_all = recursive_loop_all(voice, noisy, model, mseloss)
				else:
					recover_all = model(noisy)
				torchaudio.save(dump_dir/f"origin_{epoch}.wav", voice[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"noisy_{epoch}.wav", noise[0,:,:].to("cpu")
+	voice[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
				torchaudio.save(dump_dir/f"recover_{epoch}.wav", recover_all[0,:,:].to("cpu")
,	target_sample_rate,False,"wav","PCM_F")
			
