import torch
from torch import nn

class UnpackedGru(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, device="cpu"):
        super().__init__()
        self._reset = nn.Linear(input_size+hidden_size, hidden_size, bias=bias, device=device)
        self._map = nn.Linear(input_size+hidden_size, hidden_size, bias=bias, device=device)
        self._update = nn.Linear(input_size+hidden_size, hidden_size, bias=bias, device=device)
    
    def forward(self, state, _input):
        """
        Parameters
        ----------
        state : Tensor
            Input tensor of shape (batch_size, hidden_size)
        _input : Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        Tensor
            State tensor of shape (batch_size, hidden_size)
        """
        reset_coeff = torch.sigmoid(self._reset(torch.cat([state, _input], dim=1)))
        _reseted_state = reset_coeff*state
        mapped_input = torch.tanh(self._map(torch.cat([_reseted_state, _input], dim=1)))
        update_coeff = torch.sigmoid(self._update(torch.cat([state, _input], dim=1)))
        new_state = (1-update_coeff)*state + update_coeff*mapped_input
        return new_state
        
