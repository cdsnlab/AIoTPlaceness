# config.py
import platform
import os
class Config:
	
	if platform.system() == 'Windows':
		root_dir = 'Y:'
	else:
		root_dir = '/cdsn_nas'

	DATA_PATH = os.path.join(root_dir, 'processed')
	DATASET_PATH = os.path.join(root_dir, 'processed', 'dataset')
	CHECKPOINT_PATH = os.path.join(root_dir, 'processed', 'checkpoint')
	EMBEDDING_PATH = './embedding'
	CSV_PATH = './csv'
	MAX_SENTENCE_LEN = 513
	MAX_SEQUENCE_LEN = 10