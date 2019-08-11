import torch
import math
import os
import numpy as np

def transform_id2word(index, id2word, lang):
	if lang == "ja":
		return "".join([id2word[idx] for idx in index.data.cpu().numpy()])
	else:
		return " ".join([id2word[idx] for idx in index.data.cpu().numpy()])

def sigmoid_annealing_schedule(step, max_step, param_init=1.0, param_final=0.01, gain=0.3):
	return ((param_init - param_final) / (1 + math.exp(gain * (step - (max_step / 2))))) + param_final

def save_models(model, path, prefix, epoch):
	if not os.path.isdir(path):
		os.makedirs(path)
	model_save_path = '{}/{}_epoch_{}.pt'.format(path, prefix, epoch)
	torch.save(model, model_save_path)
