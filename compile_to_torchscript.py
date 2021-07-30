import torch

from stylegan2 import StyleGAN2


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

model_path = "./checkpoints/model_270.pt"
model = StyleGAN2(image_size=256, latent_dim=2048)	

load_data = torch.load(model_path)
model.load_state_dict(load_data['GAN'])

def infer(model, src, targ):
	(I_s, P_s, A_s), (I_t, P_t) = src, targ
	# I_s = I_s.cuda(self.rank)
	# P_s = P_s.cuda(self.rank)
	# A_s = A_s.cuda(self.rank)
	# I_t = I_t.cuda(self.rank)
	# P_t = P_t.cuda(self.rank)

	latent_dim = model.G.latent_dim
	image_size = model.G.image_size
	num_layers = model.G.num_layers


	batch_size = I_t.shape[0]

	
	# # Get encodings
	E_s = model.p_net(P_s)
	E_t = model.p_net(P_t)
	z_s_1d = model.a_net(A_s)

	z_s_def = [(z_s_1d, num_layers)]
	z_s_styles = styles_def_to_tensor(z_s_def)

	noise = image_noise(batch_size, image_size, device=rank)

	I_dash_s = model.G(z_s_styles, noise, E_s)
	I_dash_s_to_t = model.G(z_s_styles, noise, E_t)

	I_dash_s_ema = model.GE(z_s_styles, noise, E_s)
	I_dash_s_to_t_ema = model.GE(z_s_styles, noise, E_t)

	return (image_size, I_dash_s, I_dash_s_to_t,
			I_dash_s_ema, I_dash_s_to_t_ema)
