import argparse
from model.deconv_autoencoder.train import train_reconstruction
import config

CONFIG = config.Config
def main():
	parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
	# learning

	parser.add_argument('-lr', type=float, default=3e-04, help='initial learning rate')
	parser.add_argument('-weight_decay', type=float, default=1e-04, help='initial weight decay')
	parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train')
	parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')
	parser.add_argument('-lr_decay_interval', type=int, default=10,
						help='how many epochs to wait before decrease learning rate')
	parser.add_argument('-log_interval', type=int, default=100,
						help='how many steps to wait before logging training status')
	parser.add_argument('-test_interval', type=int, default=1,
						help='how many epochs to wait before testing')
	parser.add_argument('-save_interval', type=int, default=2,
						help='how many epochs to wait before saving')
	# data
	parser.add_argument('-shuffle', default=True, help='shuffle data every epoch')
	parser.add_argument('-sentence_len', type=int, default=253, help='how many tokens in a sentence')
	# model
	parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension')
	parser.add_argument('-filter_size', type=int, default=300, help='filter size of convolution')
	parser.add_argument('-filter_shape', type=int, default=5,
						help='filter shape to use for convolution')
	parser.add_argument('-latent_size', type=int, default=900, help='size of latent variable')
	parser.add_argument('-tau', type=float, default=0.01, help='temperature parameter')
	parser.add_argument('-use_cuda', action='store_true', default=True, help='whether using cuda')
	# option
	parser.add_argument('-enc_snapshot', type=str, default=None, help='filename of encoder snapshot ')
	parser.add_argument('-dec_snapshot', type=str, default=None, help='filename of encoder snapshot ')

	args = parser.parse_args()

	train_reconstruction(args, CONFIG)


if __name__ == '__main__':
	main()