from __future__ import print_function, division
import os
import csv
import re
import json
import functools
import random
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from sklearn.preprocessing import OneHotEncoder
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.base import TransformerMixin
import mongo
import copy

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_offset: torch.Tensor shape (n_i, M, 3)
      atom_pos: torch.Tensor shape (n_i, 3)
      nbr_pos: torch.Tensor shape (n_i, M, 3)
      atom_pos_idx: torch.Tensor shape (n_i, M)
      cells: torch.Tensor shape (n_i, 3, 3)
      ads_tag: torch.Tensor shape (n_i)
      fixed_base: torch.Tensor shape (n_i, 1)
      free_atom_idx: torch.Tensor shape (# free atoms)
      atom_pos_final: torch.Tensor shape (n_i, 3)
      
      target: torch.Tensor shape (n_i[free_atoms], 3)

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    batch_nbr_fea_offset: torch.Tensor shape (N, M, 3)
      Indices of the unit cell where each atom is located 
    batch_atom_pos: torch.LongTensor shape (N, 3)
      Position of each atom in the initial structures
    crystal_cell: torch.LongTensor shape (N, 3, 3)
      Unit cell vectors
    batch_ads_tag: torch.LongTensor shape (N, 1)
      Tagging adsorbate atoms
    fixed_atom_mask: torch.LongTensor shape (N, 1)
      Tagging fixed atoms
    batch_atom_pos_final: torch.LongTensor shape (N, 3)
      Position of each atom in the final/relaxed structures
    target: torch.Tensor shape (N[free_atoms], 3)
      Target value for prediction
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_nbr_fea_offset = [], [], [], []
    batch_fixed_atom_idx, batch_atom_pos = [], []
    fixed_atom_mask, batch_atom_pos_final = [], []
    crystal_cell  = []
    batch_target = []
    batch_ads_tag = []
    base_idx = 0
    
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, nbr_fea_offset, atom_pos, nbr_pos, atom_pos_idx, cells, ads_tag, fixed_base, free_atom_idx, atom_pos_final), target)\
            in enumerate(dataset_list):
            
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_atom_pos.append(atom_pos)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_offset.append(nbr_fea_offset)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_cell.append(cells)

        fixed_atom_mask.append(torch.LongTensor(fixed_base))
        batch_ads_tag.append(torch.LongTensor(ads_tag))
        
        if type(target) is not float:
            batch_target.append(torch.Tensor(target[0]).view(-1,3))
        else:
            batch_target.append(torch.Tensor([0]))
        batch_atom_pos_final.append(atom_pos_final.view(-1,3)) 
    
        base_idx += n_i

    return {'node_fea':torch.cat(batch_atom_fea, dim=0), 
            'edge_fea':torch.cat(batch_nbr_fea, dim=0), 
            'edge_idx':torch.cat(batch_nbr_fea_idx, dim=0), 
            'nbr_offset':torch.cat(batch_nbr_fea_offset, dim=0),
            'atom_pos':torch.cat(batch_atom_pos, dim=0),
            'cells': torch.cat(crystal_cell, dim=0),
            'ads_tag_base': torch.cat(batch_ads_tag),
            'fixed_atom_mask': torch.cat(fixed_atom_mask),
            'atom_pos_final': torch.cat(batch_atom_pos_final)}, torch.cat(batch_target)

class MergeDataset(torch.utils.data.Dataset):
    #Simple custom dataset to combine two datasets 
    # (one for input X, one for label y)
    def __init__(
            self,
            X,
            y,
            length=None,
    ):

        self.X = X
        self.y = copy.deepcopy(y)

        len_X = len(X)
        if y is not None:
            len_y = len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        return self._len

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, i):
        X, y = self.X, self.y
        
        if y is not None:
            yi = copy.deepcopy(y[i])
        else:
            yi = np.nan
        return X[i], yi
