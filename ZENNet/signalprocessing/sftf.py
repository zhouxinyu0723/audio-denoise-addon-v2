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
	device = sftf_window.get_device()
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
	# assertions
	assert frame.get_device() == window.get_device()
	assert len(frame.shape) == 3
	if len(window.shape) == 1:
		window = window[None,:,None]
	# sftf
	audio_in_freq = torch.fft.fft(frame*window, dim = 1)
	return audio_in_freq

def sftf_a_trace(audio: torch.Tensor, window: torch.Tensor, stride: int):
	# assertions
	assert audio.get_device() == window.get_device()
	assert len(audio.shape) == 3
	if len(window.shape) == 1:
		window = window[None,:,None]
	# sftf
	device = audio.get_device()
	sftf_output_shape = [audio.shape[0],audio.shape[1]//stride,audio.shape[1]]
	audio_in_freq = torch.zeros(sftf_output_shape, device=device)
	for i in range(audio.shape[1]//stride):
		audio_in_freq[:,i,:] = sftf_a_frame(audio[:,i*stride:(i+1)*stride,:], window)
	return audio_in_freq