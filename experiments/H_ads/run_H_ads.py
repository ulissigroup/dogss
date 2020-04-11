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

from pymatgen.io.ase import AseAtomsAdaptor

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.lbfgs import LBFGS
from cgcnn.data import StructureData, ListDataset, StructureDataTransformer, collate_pool, MergeDataset
from cgcnn.model import CrystalGraphConvNet
from torch.optim import lr_scheduler

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

from sigopt import Connection
from sigopt_sklearn.search import SigOptSearchCV


SDT_list = pickle.load(open('../../inputs/H_ads/SDT_list_new.pkl', 'rb'))
docs = pickle.load(open('../../inputs/H_ads/final_docs_new.pkl', 'rb'))


structures = SDT_list[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

SDT_list_new = []
docs_new = []
for i, sdt in enumerate(SDT_list):
    atom_pos_final = sdt[-1]
    nbr_fea_idx = sdt[2]
    nbr_fea_offset = sdt[3]
    cells = sdt[7]
    ads_idx_base = sdt[-4]
    ads_idx = np.where(ads_idx_base == 1)[0]

    nbr_pos = atom_pos_final[nbr_fea_idx]
    differ = nbr_pos - atom_pos_final.unsqueeze(1) + torch.bmm(nbr_fea_offset, cells)
    differ_sum = torch.sum(differ**2, dim=2)
    distance = torch.sqrt(differ_sum).unsqueeze(-1)
    
    if np.min(distance.numpy()) == 0:
        print('filtered idx', i)
    else:
        SDT_list_new.append(sdt)
        docs_new.append(docs[i])
        
        
target_list = np.array([sdt[-1][sdt[-2]].numpy() for sdt in SDT_list]).reshape(-1,1) #get final_pos of free atoms ONLY
SDT_list = SDT_list_new
docs = docs_new



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

def get_distance(atom_pos, nbr_fea_idx, nbr_fea_offset, cells):
    nbr_pos = atom_pos[nbr_fea_idx]
    differ = nbr_pos - atom_pos.unsqueeze(1) + torch.bmm(nbr_fea_offset, cells)
    differ_sum = torch.sum(differ**2, dim=2)
    distance = torch.sqrt(differ_sum).unsqueeze(-1)            
    return distance

differences, differences_ads, differences_non_ads_free = [], [], []
dist_free, dist_ads, dist_non_ads =[],[], []
SDT_list_new, docs_new, target_list_new = [], [], []

c = 0
max_num_nbr = 12

for i, sdt in tqdm.tqdm(enumerate(SDT_list)):
    nbr_fea_idx = sdt[2]
    nbr_fea_offset = sdt[3]
    atom_pos =sdt[4]
    cells = sdt[7]
    atom_pos_final = sdt[-1]
    free_atom_idx = sdt[-2]
    ads_idx_base = sdt[-4]
    ads_tag = np.where(ads_idx_base == 1)[0]
    non_ads_tag = np.sort(np.array(list(set(free_atom_idx.numpy()) ^ set(ads_tag))))

#     non_ads_free_base = np.arange(len(atom_pos))
#     non_ads_free_atom_idx = free_atom_idx[np.where(free_atom_idx.numpy() != ads_tag)]
    
    ads_dist = torch.sqrt(torch.sum((atom_pos-atom_pos_final)[ads_tag]**2, dim=1))
    non_ads_dist = torch.sqrt(torch.sum((atom_pos-atom_pos_final)[non_ads_tag]**2, dim=1))
    
    if torch.mean(ads_dist) < 2 and 0.04 < torch.mean(non_ads_dist) < 0.3:
        SDT_list_new.append(sdt)
        target_list_new.append(target_list[i])
        docs_new.append(docs[i])

        dist_free.append(torch.sqrt(torch.sum((atom_pos - atom_pos_final)[free_atom_idx]**2, dim=1)))
        dist_ads.append(ads_dist)
        dist_non_ads.append(non_ads_dist)

        bond_distance = get_distance(atom_pos, nbr_fea_idx, nbr_fea_offset, cells)
        final_distance = get_distance(atom_pos_final, nbr_fea_idx, nbr_fea_offset, cells)
        N, M, C = bond_distance.shape
        bond_distance = bond_distance #* fake_nbr.float().expand(N, M, C)    
        final_distance = final_distance #* fake_nbr.float().expand(N, M, C) 

        c += len(atom_pos) * max_num_nbr
    #     differences.append((torch.sqrt(torch.tensor(10.))*final_distance - torch.sqrt(torch.tensor(10.))*bond_distance).view(-1))
        differences.append((final_distance - bond_distance).view(-1))
        differences_ads.append((final_distance - bond_distance).view(-1)[ads_tag])
        differences_non_ads_free.append((final_distance - bond_distance).view(-1)[non_ads_tag])

differences = torch.cat(differences)**2
differences_ads = torch.cat(differences_ads)**2
differences_non_ads_free = torch.cat(differences_non_ads_free)**2

# differences = torch.clamp(differences, min=1e-8)
assert c == len(differences)
# dist_err = torch.mean(torch.abs(differences))
# torch.log(dist_err)
# dist_err = torch.mean(torch.sqrt(differences))
dist_err = torch.mean(differences)
dist_err_ads = torch.mean(differences_ads)
dist_err_non_ads_free = torch.mean(differences_non_ads_free)

print('dataset length', len(SDT_list_new))
print('bond_distance:',dist_err, dist_err_ads, dist_err_non_ads_free)
print('x:',torch.mean(torch.cat(dist_free)), torch.mean(torch.cat(dist_ads)), torch.mean(torch.cat(dist_non_ads)))



target_list_new = np.array(target_list_new)

SDT_training, SDT_test, target_training, target_test, docs_training, docs_test \
= train_test_split(SDT_list_new, target_list_new, docs_new, test_size=0.1, random_state=42)


from torch.optim import lr_scheduler
import torch.optim as optim


train_test_splitter = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
# train_test_splitter = ShuffleSplit(n_splits=1, test_size=0.1, random_state=22)


batchsize = 20
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
    lr=0.028446171826811632,
    max_epochs= 200,
    module__energy_mode="Harmonic", #["Harmonic", "Morse", "LJ"], Default = "Harmonic"
    module__atom_fea_len=128, #46,
    module__h_fea_len=105,
    module__h_fea_len_dist=71,
    module__h_fea_len_const=71,
#     module__h_fea_len_D=(3,256),
    module__n_conv=9, #8
    module__n_h_dist=0,
    module__n_h_const=0,
    optimizer__weight_decay=0.000045399929762484854,
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


if __name__ == '__main__':
    net.initialize()
    net.fit(SDT_training, target_training)
