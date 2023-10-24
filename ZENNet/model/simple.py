import torch
from torch import nn, reshape
from torch import device as torch_device

class Simple(nn.Module):
	"""
	Simple model
	use mlp to do denoise	
	"""
	def __init__(self, samples, chunk_size, channels, device):
		super().__init__()
		self.chunk_size = chunk_size
		self.channels = channels
		self.linear = nn.Linear(self.chunk_size*self.channels, self.chunk_size*self.channels, bias=False, device = device)
	
	def forward(self, state, _input):
		"""
		Parameters
		----------
		x : Tensor
		    Input tensor of shape (batch_size, samples, channels)
		state : Tensor
		    Input tensor of shape (batch_size, hidden_dim, channels)
		Returns
		-------
		Tensor
		    State tensor of shape (batch_size, hidden_dim, channels)
		Tensor
		    Output tensor of shape (batch_size, samples, channels)
		"""
		if len(_input.shape)==3:
			batch_size = _input.shape[0]
		else:
			batch_size = 1
		shape_saved = _input.shape
		std = torch.std(_input)
		_input = _input/std
		_res = reshape(_input, (batch_size,-1))
		_res = self.linear(_res)
		_res = reshape(_res, shape_saved)	
		return None, (_input+_res)*std

	@property
	def is_recursive(self):
		return True
