import torch
from torch import nn, reshape
from torch import device as torch_device

class Simple_fft(nn.Module):
	"""
	Simple model
	use mlp to do denoise	
	"""
	def __init__(self, samples, chunk_size, channels, device):
		super().__init__()
		self.chunk_size = chunk_size
		self.channels = channels
		self.device = device
		self.linear = nn.Linear(self.chunk_size*self.channels, self.chunk_size*self.channels, bias=True, device = device)
		self.linear2 = nn.Linear(self.chunk_size*self.channels, self.chunk_size*self.channels, bias=True, device = device)
		self.linear3 = nn.Linear(self.chunk_size*self.channels, self.chunk_size*self.channels, bias=True, device = device)
		#self.scale = torch.nn.Parameter(torch.Tensor(1), requires_grad = True)
		#self.scale.data.fill_(1)
	
	def forward(self, state, _input):
		"""
		Parameters
		----------
		_input : Tensor
		  Input tensor of shape (batch_size, samples, channels)
			the amplitute of freq bins
		state : Tensor
		  Input tensor of shape (batch_size, hidden_dim, channels)
		Returns
		-------
		Tensor
		  State tensor of shape (batch_size, hidden_dim, channels)
		Tensor
		  Output tensor of shape (batch_size, samples, channels)
			Between 0 and 1, intensity of suppression in freq bins 
		"""
		
		if len(_input.shape)==3:
			batch_size = _input.shape[0]
		else:
			batch_size = 1
		shape_saved = _input.shape
		std = torch.std(_input)
		_input = _input/std
		_input = reshape(_input, (batch_size,-1))
		_input = self.linear(_input)
		_input = nn.functional.relu(_input)
		_input = self.linear2(_input)
		_input = nn.functional.relu(_input)
		_input = self.linear3(_input)
		_input = nn.functional.relu(_input)
		_input = reshape(_input, shape_saved)	
		#print(self.scale.detach().clone().to("cpu"))
		return None, torch.sigmoid(_input)#*torch.exp(self.scale)

	@property
	def is_recursive(self):
		return True
