'''
Config for LSTM model 
'''
import pathlib
# predictions
from os import makedirs

DEFAULT_MODEL = 'june2019'
PREDICTIONS_DIR = pathlib.Path('./predictions')
MEL_CACHE_DIR = pathlib.Path('./mel_cache')

try:
    makedirs(PREDICTIONS_DIR)
except Exception:
    pass
try:
    makedirs(MEL_CACHE_DIR)
except Exception:
    pass

# jamendo
JAMENDO_DIR = pathlib.Path('../jamendo/')  # path to jamendo dataset
JAMENDO_LABEL_DIR = pathlib.Path('../jamendo/labels/ ')  # path to jamendo label
MEL_JAMENDO_DIR = pathlib.Path('../jamendo/mel/')

# Medleydb
MDB_VOC_DIR = '/media/bach1/dataset/MedleyDB/'  # path to medleydb vocal containing song
MDB_LABEL_DIR = '../mdb_voc_label/'

# vibrato 
VIB_DIR = '../sawtooth_200/songs/'
MEL_VIB_DIR = '../sawtooth_200/leglaive_mel_dir/'

# -- Audio processing parameters --#
SR = 16000
N_FFT1 = 4096
N_HOP1 = 2048
N_FFT2 = 512
N_HOP2 = 256

RNN_INPUT_SIZE = 218  # 7sec/(256/16000)
RNN_OVERLAP = 10
N_MELS = 40
N_MFCC = 40

# -- model parameters --#
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
INPUT_SIZE = 80  # 40 harmonic + 40 percussive
NUM_EPOCHS = 100
THRESHOLD = 0.5
