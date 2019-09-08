# config.py
import platform
import os
class Config:
	
	if platform.system() == 'Windows':
		root_dir = 'Y:'
	else:
		root_dir = '/cdsn-nas'

	TARGET_PATH = os.path.join(root_dir, 'placeness')
	DATA_PATH = os.path.join(root_dir, 'processed')
	DATASET_PATH = os.path.join('./data', 'dataset')
	CHECKPOINT_PATH = os.path.join(root_dir, 'processed', 'checkpoint')
	EMBEDDING_PATH = './embedding'
	CSV_PATH = './csv'
	MAX_SENTENCE_LEN = 257
	MIN_WORD_COUNT = 5
	MAX_SEQUENCE_LEN = 10
	SVG_PATH = './svg'