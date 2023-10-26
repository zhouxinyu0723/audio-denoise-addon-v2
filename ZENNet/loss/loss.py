import torch
from torch import nn

from ZENNet.signalprocessing.sftf import hann_window

mseloss = nn.MSELoss()	# mean square error

def sftf_loss_prepare(recover: torch.Tensor, voice: torch.Tensor) -> torch.Tensor:
    assert recover.shape == voice.shape
    assert len(recover.shape) == 3
    device = recover.get_device()
    sftf_window = hann_window(recover.shape[1]).to(device)[None,:,None]
    recover_F_loss = torch.fft.fft(recover*sftf_window, dim = 1)
    recover_F_loss_abs = recover_F_loss.abs()
    recover_F_loss_angle = recover_F_loss.angle()
    voice_F_loss = torch.fft.fft(voice*sftf_window, dim = 1)
    voice_F_loss_abs = voice_F_loss.abs()
    voice_F_loss_angle = voice_F_loss.angle()
    return recover_F_loss_abs, recover_F_loss_angle, voice_F_loss_abs, voice_F_loss_angle