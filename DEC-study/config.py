# config.py
import platform
import os


class Config:
    # preprocess config
    IMAGE_SIZE = 448  # scale shorter end of image to this size and centre crop
    OUTPUT_SIZE = IMAGE_SIZE // 32  # size of the feature maps after processing through a network
    OUTPUT_FEATURES = 2048  # number of feature maps thereof
    CENTRAL_FRACTION = 0.875  # only take this much of the centre when scaling and centre cropping
    CSV_PATH = './csv'
    MAX_SENTENCE_LEN = 257
    MIN_WORD_COUNT = 5
    N_CLASSES = 12
    SVG_PATH = './svg'
    DATA_WORKERS = 4 # cause error in python 3.6
    RESULT_PATH = './result'