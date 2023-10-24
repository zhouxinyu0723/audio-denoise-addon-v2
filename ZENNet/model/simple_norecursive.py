from torch import nn, reshape
from torch import device as torch_device

class Simple_NR(nn.Module):
	"""
	Simple model
	use mlp to do denoise	
	"""
	def __init__(self, samples, chunk_size,channels, device):
		super().__init__()
		self.samples = samples
		self.channels = channels
		self.linear = nn.Linear(self.samples*self.channels, self.samples*self.channels, bias=False, device = device)
	
	def forward(self, _input):
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
		_input = reshape(_input, (batch_size,-1))
		_input = self.linear(_input)
		_input = reshape(_input, shape_saved)	
		return _input

	@property
	def is_recursive(self):
		return False
