import os
import sys
sys.path.insert(0,'/home/junwoony/Desktop/DOGSS/')

import numpy as np
# %env CUDA_DEVICE_ORDER=PCI_BUS_ID
# %env CUDA_VISIBLE_DEVICES=1
# %env CUDA_LAUNCH_BLOCKING=1

import mongo
import time
import pickle
import random
import numpy as np
import tqdm
import copy
import matplotlib.pyplot as plt
import multiprocess as mp
import seaborn as sns
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.lbfgs import LBFGS
from cgcnn.data import StructureData, ListDataset, StructureDataTransformer, collate_pool, MergeDataset
from cgcnn.model import CrystalGraphConvNet

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import get_scorer
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint, LoadInitState #needs skorch 0.4.0, conda-forge version at 0.3.0 doesn't cut it
import skorch.callbacks.base
from skorch.dataset import CVSplit
from skorch.callbacks.lr_scheduler import WarmRestartLR, LRScheduler

from utils.adamwr.adamw import AdamW
from utils.adamwr.cosine_scheduler import CosineLRWithRestarts


from sigopt_sklearn.search import SigOptSearchCV


SDT_list = pickle.load(open('../../inputs/surfaces/SDT_list_new.pkl', 'rb'))
docs = pickle.load(open('../../inputs/surfaces/final_docs.pkl', 'rb'))

structures = SDT_list[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

target_list = np.array([sdt[-1][sdt[-2]].numpy() for sdt in SDT_list]).reshape(-1,1) #get final_pos of free atoms ONLY

SDT_training, SDT_test, target_training, target_test, docs_training, docs_test \
= train_test_split(SDT_list, target_list, docs, test_size=0.1, random_state=42)

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device='cpu'

#Make a checkpoint to save parameters every time there is a new best for validation lost
cp = Checkpoint(monitor='valid_loss_best',fn_prefix='valid_best_')

#Callback to load the checkpoint with the best validation loss at the end of training

class train_end_load_best_valid_loss(skorch.callbacks.base.Callback):
    def on_train_end(self, net, X, y):
        net.load_params('valid_best_params.pt')
        
load_best_valid_loss = train_end_load_best_valid_loss()
print('device:', device)


def diff(sdt, target):
    fixed_base = sdt[-3]
    free_atom_idx = np.where(fixed_base == 0)[0]
    free_atom_idx = torch.LongTensor(free_atom_idx)   
    diff = np.sum(((target[0] - sdt[4].numpy()[free_atom_idx]))**2.,axis=1)**0.5 
    return diff
print('Initial loss')
print(np.mean(np.abs(np.concatenate([diff(sdt, target) for sdt,target in zip(SDT_test, target_test)]))))

from torch.optim import lr_scheduler
import torch.optim as optim


train_test_splitter = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

batchsize = 17
# warm restart scheduling from https://arxiv.org/pdf/1711.05101.pdf
# LR_schedule = LRScheduler(CosineLRWithRestarts, batch_size=batchsize, epoch_size=len(SDT_training), restart_period=10, t_mult=1.2)

#### For Sigopt
LR_schedule = LRScheduler("MultiStepLR", milestones=[100], gamma=0.1)

#############
# To extract intermediate features, set the forward takes only the first return value to calculate loss
class MyNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, **kwargs):
        y_pred = y_pred[0] if isinstance(y_pred, tuple) else y_pred  # discard the 2nd output
        differ=torch.sum((y_pred-y_true.cuda())**2.0,dim=1)
        if torch.nonzero(differ).shape[0] != differ.shape[0]:
            print('zero sqrt for Loss')
#             zero_idx = (differ == 0).nonzero()
#             differ[zero_idx] = 1e-6
        differ = torch.clamp(differ, min=1e-8)

        return torch.mean(torch.sqrt(differ))


net = MyNet(
    CrystalGraphConvNet,
    module__orig_atom_fea_len = orig_atom_fea_len,
    module__nbr_fea_len = nbr_fea_len,
    batch_size=batchsize, #214
    module__classification=False,
    lr=0.0393415,
    max_epochs= 200,
    module__energy_mode="Harmonic", #["Harmonic", "Morse", "LJ"], Default = "Harmonic"
    module__atom_fea_len=236, #46,
    module__h_fea_len=6,
    module__h_fea_len_dist=4,
    module__h_fea_len_const=4,
#     module__h_fea_len_D=(3,256),
    module__n_conv=12, #8
    module__n_h_dist=0,
    optimizer__weight_decay=0.0000454,
    module__n_h_const=0,
#     module__n_h_D=(1,12),
#     module__max_num_nbr=12, #9
#     module__opt_step_size=(0.1,0.7), #0.3
    module__min_opt_steps=30,
    module__max_opt_steps=150,
    module__momentum=0.8,
    optimizer=AdamW,
    iterator_train__pin_memory=True,
    iterator_train__num_workers=0,
    iterator_train__shuffle=True,
    iterator_train__collate_fn = collate_pool,
    iterator_valid__pin_memory=True,
    iterator_valid__num_workers=0,
    iterator_valid__collate_fn = collate_pool,
    device=device,
#     criterion=torch.nn.MSELoss,
    criterion=torch.nn.L1Loss,
    dataset=MergeDataset,
    train_split = CVSplit(cv=train_test_splitter),
    callbacks=[cp,LR_schedule , load_best_valid_loss],

)

net.initialize()
net.fit(SDT_training, target_training)
