import torch
from torch import nn, reshape

from ZENNet.model.unpackedGru import UnpackedGru

class SimpleStftGru(nn.Module):
	"""
	Simple model
	use mlp to do denoise	
	"""
	def __init__(self, samples, chunk_size, channels, device):
		super().__init__()
		self.chunk_size = chunk_size
		self.channels = channels
		self.device = device
		self.input_map = nn.Linear(self.chunk_size*self.channels, self.chunk_size*self.channels, bias=True, device = device)
		self.unpacked_gru = UnpackedGru(self.chunk_size*self.channels, self.chunk_size*self.channels, bias=True, device = device)
		self.state_input2output = nn.Linear(2*self.chunk_size*self.channels, self.chunk_size*self.channels, bias=True, device = device)	
	
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
		if state == None:
			state = torch.zeros([batch_size, self.chunk_size*self.channels]).to(self.device)
		shape_saved = _input.shape
		_input = reshape(_input, (batch_size,-1))
		_input_mapped = nn.functional.relu(self.input_map(_input))

		_output = torch.sigmoid(self.state_input2output(torch.cat([state, _input_mapped], dim=1)))
		_output = reshape(_output, shape_saved)	

		_state = self.unpacked_gru(state, _input)

		return _state, _output

	@property
	def is_recursive(self):
		return True
