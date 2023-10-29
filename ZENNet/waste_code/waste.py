#fig, ax = plt.subplots(3,1)
#ax[0].plot(np.linspace(0, 100, chunk_size), _windowDIV_template.numpy())
#ax[1].plot(np.linspace(0, 100, chunk_size), _windowMUL.numpy())
#ax[2].plot(np.linspace(0, 100, target_samples), windowMUL.numpy())
#plt.show()



def recursive_loop(voice, noisy, model):
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
	Tensor
		Output tensor of shape (batch_size, target_samples, channels)
		The recoverd audio
	"""
	hidden_state = None
	batch_size = voice.shape[0]
	recover_all = torch.zeros([batch_size,target_samples, target_channels]).to(device)
	avg_hidden = 0
	avg_hidden_std = 0
	for i in range((target_samples-window_len)//window_stride +1):
		start = i*window_stride
		end = start+window_len
		with torch.no_grad():
			noisy_T = noisy[:,start:end,:]*sftf_window
			noisy_F = torch.fft.fft(noisy_T, dim = 1)
		hidden_state, suppression_hint = model(hidden_state, noisy_F.abs())
		recover_abs = noisy_F.abs()*suppression_hint
		recover_angle = noisy_F.angle()
		recover_F = torch.polar(recover_abs, recover_angle)
		recover_T = torch.fft.ifft(recover_F, dim = 1)
		recover_all[:,start: end,:] += recover_T.real*isftf_window
		avg_hidden += torch.mean(hidden_state).detach().to("cpu").numpy().item()
		avg_hidden_std += torch.std(hidden_state).detach().to("cpu").numpy().item()
	avg_hidden /= (target_samples-window_len)//window_stride +1
	print(avg_hidden)
	print(avg_hidden_std)
	recover_all *= recover_coeff_all
	# TODO wrap the following into a new loss function
	#print(model.scale.detach())
	#print(loss_func(voice[:,start:end,:], recover_all[:,start:end,:])*100)
	#print(loss_func(noisy[:,start:end,:], voice[:,start:end,:])*100)
	#return loss_func(voice[:,start:end,:], recover_all[:,start:end,:])

	loss_window_size = window_len*randint(1, min(5, target_samples//window_len))
	start = randint(max(0,target_samples-loss_window_size-window_stride-2*window_len), target_samples-loss_window_size-window_len)
	end = start + loss_window_size

	recover_F_loss_abs, recover_F_loss_angle, voice_F_loss_abs, voice_F_loss_angle = sftf_loss_prepare(recover_all[:,start:end,:], voice[:,start:end,:])
	avg_db = torch.mean(torch.log(voice_F_loss_abs+1e-7),  1, True)
	loss = mseloss(torch.clamp(torch.log(recover_F_loss_abs+1e-7), min=avg_db-20), torch.clamp(torch.log(voice_F_loss_abs+1e-7), min=avg_db-20))
	#loss += 0.3*loss_func(recover_F_loss_angle, voice_F_loss_angle) 
	print(torch.mean(suppression_hint).to("cpu"))
	print(loss.to("cpu")*100)

	noisy_F_loss_abs, noisy_F_loss_angle, voice_F_loss_abs, voice_F_loss_angle = sftf_loss_prepare(noisy[:,start:end,:], voice[:,start:end,:])
	loss_ref = mseloss(torch.clamp(torch.log(noisy_F_loss_abs+1e-7), min=avg_db-20), torch.clamp(torch.log(voice_F_loss_abs+1e-7), min=avg_db-20))
	print(loss_ref.to("cpu")*100)
	print(loss/loss_ref*100)

	return loss, recover_all