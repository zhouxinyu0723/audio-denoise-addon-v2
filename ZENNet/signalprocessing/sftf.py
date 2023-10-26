import math
import torch

def hann_window(length: int):
	"""
	Parameters
	----------
	length : int
		Length of the window
	
	Returns
	----------
	Tensor
		Window of shape (length,), no zero value
	"""
	cache = {}
	if length in cache:
		return cache[length]
	else:
		angle_speed = math.pi/length
		cache[length] = torch.square(torch.sin((torch.arange(length)+0.5)*angle_speed)) 
		return cache[length]

def isftf_recover_coef(sftf_window: torch.Tensor, isftf_window: torch.Tensor, window_stride: int):
	"""
	Parameters
	----------
	f2t_window : Tensor
		Window of shape (window_size,), multiplied before STFT
	t2f_window : Tensor
		Window of shape (window_size,), multiplied after ISTFT
	window_stride : int
		Stride of the window

	Returns
	----------
	Tensor
		Window of shape (window_size,), multiplied after ISTFT merge
	"""
	assert sftf_window.get_device() == isftf_window.get_device()
	device = "cuda" if sftf_window.get_device() != -1 else "cpu"
	assert sftf_window.shape == isftf_window.shape
	assert len(sftf_window.shape) == 1
	window_size = sftf_window.shape[0]
	assert window_size > window_stride
	recover_devider = torch.zeros(sftf_window.shape, device=device)
	tmp = sftf_window*isftf_window
	for i in range(window_size//window_stride):
		if i==0:
			recover_devider += tmp
		else:
			recover_devider[i*window_stride:] += tmp[:window_size-i*window_stride]
			recover_devider[:window_size-i*window_stride] += tmp[i*window_stride:]
	return 1/recover_devider

def sftf_a_frame(frame: torch.Tensor, window: torch.Tensor):
	"""
	frame : Tensor
		Input tensor of shape (batch_size, samples, channels)
	window : Tensor
		Window of shape (window_size,) or of shape (1, window_size, 1)
		, multiplied before STFT
	"""
	# assertions
	assert frame.get_device() == window.get_device()
	assert len(frame.shape) == 3
	if len(window.shape) == 1:
		window = window[None,:,None]
	# sftf
	audio_in_freq = torch.fft.fft(frame*window, dim = 1)
	return audio_in_freq

def sftf_a_trace(audio: torch.Tensor, window: torch.Tensor, stride: int):
	"""
	audio : Tensor
		Input tensor of shape (batch_size, samples, channels)
	window : Tensor
		Window of shape (window_size,) or of shape (1, window_size, 1)
		, multiplied before STFT
	stride : int
		Stride of the window

	Returns
	----------
	Tensor
		Window of shape (frames, batch_size, window_size, channels)
	"""
	# assertions
	assert audio.get_device() == window.get_device()
	assert len(audio.shape) == 3
	if len(window.shape) == 1:
		window = window[None,:,None]
	# sftf
	device = "cuda" if audio.get_device() != -1 else "cpu"
	window_size = window.shape[1]
	batch_size, samples, channels = audio.shape
	frames = (audio.shape[1]-window_size)//stride+1
	sftf_output_shape = [frames,batch_size,window_size,channels]
	audio_in_freq = torch.zeros(sftf_output_shape, device=device, dtype=torch.cfloat)
	for i in range(frames):
		audio_in_freq[i,:,:,:] = sftf_a_frame(audio[:,i*stride:i*stride+window_size,:], window)[None,:,:,:]
	return audio_in_freq

def isftf_a_trace(audio_in_freq: torch.Tensor, window: torch.Tensor, recover_coeff: torch.Tensor, stride: int):
	"""
	audio_in_freq : Tensor
		Input tensor of shape (frames, batch_size, window_size, channels)
	window : Tensor
		Window of shape (window_size,) or of shape (1, window_size, 1)
		, multiplied after ISTFT
	recover_coeff : Tensor
		Window of shape (window_size,) or of shape (1, window_size, 1)
		, multiplied after ISTFT merge
	stride : int
		Stride of the window
	
	Returns
	----------
	Tensor
		Window of shape (batch_size, samples, channels)
	"""
	# assertions
	assert audio_in_freq.get_device() == window.get_device()
	assert len(audio_in_freq.shape) == 4
	if len(window.shape) == 1:
		window = window[None,:,None]
	assert len(window.shape) == 3
	if len(recover_coeff.shape) == 1:
		recover_coeff = recover_coeff[None,:,None]
	assert len(recover_coeff.shape) == 3
	assert recover_coeff.shape[1] == window.shape[1]
	# sftf
	frames, batch_size, window_size, channels = audio_in_freq.shape
	samples = (frames-1)*stride+window_size
	assert samples % window_size == 0	# length of recovered audio are multiple of window_size
	device = "cuda" if audio_in_freq.get_device() != -1 else "cpu"
	isftf_output_shape = [batch_size,samples,channels]
	audio = torch.zeros(isftf_output_shape, device=device, dtype=torch.cfloat)
	for i in range(frames):
		audio[:,i*stride:i*stride+window_size,:] += torch.fft.ifft(torch.squeeze(audio_in_freq[i,:,:,:]), dim = 1)*window
	for i in range(samples // window_size):
		audio[:,i*window_size:(i+1)*window_size,:] *= recover_coeff
	return audio


def main():
	a = torch.randn(5,120,2)
	a[0,:,0] = torch.linspace(0, 1, 120)
	w = hann_window(30)
	stride = 10
	a_T = sftf_a_trace(a, w, stride)
	a_F = isftf_a_trace(a_T, w, isftf_recover_coef(w, w, stride), stride)

	from matplotlib import pyplot as plt
	import numpy as np
	fig, ax = plt.subplots(2,1)
	ax[0].plot(np.linspace(0, 100, 120), a.numpy()[0,:,0])
	ax[1].plot(np.linspace(0, 100, 120), a_F.numpy()[0,:,0])
	plt.show()

if __name__ == "__main__":
	main()