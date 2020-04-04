# config.py
import platform
import os


class Config:
    if platform.system() == 'Windows':
        ROOT_DIR = 'Y:'
    else:
        ROOT_DIR = '/cdsn-nas'

    DATA_PATH = os.path.join(ROOT_DIR, 'placeness')
    PROCESSED_PATH = os.path.join('/ssdmnt', 'processed')
    EMBEDDING_PATH = './embedding'
    CSV_PATH = './csv'
    MAX_SENTENCE_LEN = 257
    MIN_WORD_COUNT = 5
    MAX_SEQUENCE_LEN = 10
    SVG_PATH = './svg'
    RESULT_PATH = './result'