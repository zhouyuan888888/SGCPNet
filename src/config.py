import numpy as np
import os

TRAIN_DIR = './data/cityscapes/'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = './data/train_list.txt'
VAL_LIST = './data/val_list.txt'
SHORTER_SIDE = 350
CROP_SIZE =  1024
NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD

BATCH_SIZE = 36
NUM_WORKERS = 16
NUM_CLASSES = 19
LOW_SCALE = 0.5
HIGH_SCALE = 2.0
IGNORE_LABEL = 255


ENC_PRETRAINED = True
EVALUATE =True
FREEZE_BN = False
NUM_SEGM_EPOCHS = 350
PRINT_EVERY = 10
RANDOM_SEED = 42

SNAPSHOT_DIR = './'
LOG_DIR = SNAPSHOT_DIR + "log/"
CKPT_PATH = '../checkpoint/cityscapes/checkpoint3.pth.tar'

LR_ENC = 0.3
LR_DEC = 0.3
MOM_ENC = 0.9
MOM_DEC = 0.9
WD_ENC = 1e-5
WD_DEC = 1e-5
OPTIM_DEC = 'sgd'


if not os.path.exists("/".join(SNAPSHOT_DIR.split("/")[:-1])):
    os.mkdir("/".join(SNAPSHOT_DIR.split("/")[:-1]))
if not os.path.exists(SNAPSHOT_DIR):
    os.mkdir(SNAPSHOT_DIR)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

