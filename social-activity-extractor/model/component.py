import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# simply define a silu function
def silu(input):
	'''
	Applies the Sigmoid Linear Unit (SiLU) function element-wise:
		SiLU(x) = x * sigmoid(x)
	'''
	return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
	'''
	Applies the Sigmoid Linear Unit (SiLU) function element-wise:
		SiLU(x) = x * sigmoid(x)
	Shape:
		- Input: (N, *) where * means, any number of additional
		  dimensions
		- Output: (N, *), same shape as the input
	References:
		-  Related paper:
		https://arxiv.org/pdf/1606.08415.pdf
	Examples:
		>>> m = silu()
		>>> input = torch.randn(2)
		>>> output = m(input)
	'''
	def __init__(self):
		'''
		Init method.
		'''
		super().__init__() # init the base class

	def forward(self, input):
		'''
		Forward pass of the function.
		'''
		return silu(input) # simply apply already implemented SiLU

class Maxout(nn.Module):
	def __init__(self, pool_size):
		super().__init__()
		self._pool_size = pool_size

	def forward(self, x):
		assert x.shape[1] % self._pool_size == 0, \
			'Wrong input at dim 1 size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
		m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
		return m

class PTanh(nn.Module):
	'''
	Applies the Penalized Tanh (PTanh) function element-wise:
		ptanh(x) = tanh(x) if x > 0 or p * tanh(x)
	Shape:
		- Input: (N, *) where * means, any number of additional
		  dimensions
		- Output: (N, *), same shape as the input
	References:
		-  Related paper:
		https://arxiv.org/pdf/1606.08415.pdf
	Examples:
		>>> m = PTanh()
		>>> input = torch.randn(2)
		>>> output = m(input)
	'''
	def __init__(self, penalty=0.25):
		'''
		Init method.
		'''
		super().__init__() # init the base class
		self.penalty = penalty # init penalty

	def forward(self, input):
		'''
		Forward pass of the function.
		'''
		return torch.where(input > 0, torch.tanh(input), self.penalty * torch.tanh(input))

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

class AdamW(Optimizer):
	"""Implements Adam algorithm.
	It has been proposed in `Adam: A Method for Stochastic Optimization`_.
	Arguments:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		lr (float, optional): learning rate (default: 1e-3)
		betas (Tuple[float, float], optional): coefficients used for computing
			running averages of gradient and its square (default: (0.9, 0.999))
		eps (float, optional): term added to the denominator to improve
			numerical stability (default: 1e-8)
		weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
		amsgrad (boolean, optional): whether to use the AMSGrad variant of this
			algorithm from the paper `On the Convergence of Adam and Beyond`_
	.. _Adam\: A Method for Stochastic Optimization:
		https://arxiv.org/abs/1412.6980
	.. _On the Convergence of Adam and Beyond:
		https://openreview.net/forum?id=ryQu7f-RZ
	"""

	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
				 weight_decay=0, amsgrad=False):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
		defaults = dict(lr=lr, betas=betas, eps=eps,
						weight_decay=weight_decay, amsgrad=amsgrad)
		super(AdamW, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(AdamW, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('amsgrad', False)

	def step(self, closure=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
				amsgrad = group['amsgrad']

				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state['step'] = 0
					# Exponential moving average of gradient values
					state['exp_avg'] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					state['exp_avg_sq'] = torch.zeros_like(p.data)
					if amsgrad:
						# Maintains max of all exp. moving avg. of sq. grad. values
						state['max_exp_avg_sq'] = torch.zeros_like(p.data)

				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				if amsgrad:
					max_exp_avg_sq = state['max_exp_avg_sq']
				beta1, beta2 = group['betas']

				state['step'] += 1

				# if group['weight_decay'] != 0:
				#     grad = grad.add(group['weight_decay'], p.data)

				# Decay the first and second moment running average coefficient
				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
				if amsgrad:
					# Maintains the maximum of all 2nd moment running avg. till now
					torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
					# Use the max. for normalizing running avg. of gradient
					denom = max_exp_avg_sq.sqrt().add_(group['eps'])
				else:
					denom = exp_avg_sq.sqrt().add_(group['eps'])

				bias_correction1 = 1 - beta1 ** state['step']
				bias_correction2 = 1 - beta2 ** state['step']
				step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

				# p.data.addcdiv_(-step_size, exp_avg, denom)
				p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

		return loss

class ConstantLRSchedule(LambdaLR):
	""" Constant learning rate schedule.
	"""
	def __init__(self, optimizer, last_epoch=-1):
		super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
	""" Linear warmup and then constant.
		Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
		Keeps learning rate schedule equal to 1. after warmup_steps.
	"""
	def __init__(self, optimizer, warmup_steps, last_epoch=-1):
		self.warmup_steps = warmup_steps
		super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

	def lr_lambda(self, step):
		if step < self.warmup_steps:
			return float(step) / float(max(1.0, self.warmup_steps))
		return 1.


class WarmupLinearSchedule(LambdaLR):
	""" Linear warmup and then linear decay.
		Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
		Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
	"""
	def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.t_total = t_total
		super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

	def lr_lambda(self, step):
		if step < self.warmup_steps:
			return float(step) / float(max(1, self.warmup_steps))
		return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
	""" Linear warmup and then cosine decay.
		Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
		Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
		If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
	"""
	def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.t_total = t_total
		self.cycles = cycles
		super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

	def lr_lambda(self, step):
		if step < self.warmup_steps:
			return float(step) / float(max(1.0, self.warmup_steps))
		# progress after warmup
		progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
		return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
	""" Linear warmup and then cosine cycles with hard restarts.
		Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
		If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
		learning rate (with hard restarts).
	"""
	def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.t_total = t_total
		self.cycles = cycles
		super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

	def lr_lambda(self, step):
		if step < self.warmup_steps:
			return float(step) / float(max(1, self.warmup_steps))
		# progress after warmup
		progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
		if progress >= 1.0:
			return 0.0
		return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

	# Scaler: we can adapt this if we do not want the triangular CLR
	scaler = lambda x: 1/x

	# Lambda function to calculate the LR
	lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

	# Additional function to see where on the cycle we are
	def relative(it, stepsize):
		cycle = math.floor(1 + it / (2 * stepsize))
		x = abs(it / stepsize - 2 * cycle + 1)
		return max(0, (1 - x)) * scaler(cycle)

	return lr_lambda

class last_layer(nn.Module):
	def __init__(self, latent_size):
		super(last_layer, self, in_feature, latent_size).__init__()

		self.fc = nn.Linear(in_feature, latent_size)
		
	def forward(self, x):
		normalized_x = F.normalize(x, p=2, dim=1)
		return normalized_x

class ResNet50Encoder(nn.Module):
	def __init__(self, latent_size, pretrained=True, in_feature=2048):
		super(ResNet50Encoder, self).__init__()
		
		self.embedding_model = models.resnet50(pretrained=pretrained)
		self.embedding_model.fc = nn.Linear(in_feature, latent_size)

	def forward(self,x):
			
		x = self.embedding_model(x)
		return x

class ResNet50Decoder(nn.Module):
	def __init__(self, latent_size):
		super(ResNet50Decoder,self).__init__()

		self.dfc3 = nn.Linear(latent_size, 2048)
		self.bn3 = nn.BatchNorm1d(2048)
		self.dfc2 = nn.Linear(2048, 4096)
		self.bn2 = nn.BatchNorm1d(4096)
		self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
		self.bn1 = nn.BatchNorm1d(256*6*6)
		self.upsample1=nn.Upsample(scale_factor=2)
		self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
		self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
		self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
		self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
		self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self,x):
		
		x = self.dfc3(x)
		x = self.relu(self.bn3(x))
		x = self.dfc2(x)
		x = self.relu(self.bn2(x))
		x = self.dfc1(x)
		x = self.relu(self.bn1(x))
		x = x.view(x.size()[0], 256, 6, 6)
		x = self.upsample1(x)
		x = self.dconv5(x)
		x = self.relu(x)
		x = self.relu(self.dconv4(x))
		x = self.relu(self.dconv3(x))
		x = self.upsample1(x)
		x = self.dconv2(x)
		x = self.relu(x)
		x = self.upsample1(x)
		x = self.dconv1(x)
		x = self.sigmoid(x)

		return x

class ImgseqComponentAutoEncoder(nn.Module):
	def __init__(self, encoder, decoder):
		super(ImgseqComponentAutoEncoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		h = self.encoder(x)
		x_hat = self.decoder(h)
		return x_hat