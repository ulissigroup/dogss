from __future__ import print_function, division

import torch
import pickle

import torch.nn as nn
import numpy as np
from ase.geometry import wrap_positions
import copy
# from data import GaussianDistance

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.LeakyReLU()
        
    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, h_fea_len_dist=128, h_fea_len_const=128, h_fea_len_D=128, n_h_dist=1, n_h_const=1, n_h_D = 1,max_num_nbr=12, momentum=0.9, classification=False, max_opt_steps=300, min_opt_steps=10, opt_step_size=0.3, radius = 6, step_size=0.4, energy_mode="Harmonic"):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        
        self.energy_mode = energy_mode
        self.max_num_nbr = max_num_nbr
        self.momentum = momentum
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        self.gdf = GaussianDistance(dmin=0, dmax=radius, step_size=step_size)

        
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        self.conv_to_fc = nn.Linear(2*atom_fea_len + nbr_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.LeakyReLU()
        
        self.bond_distance = nn.Linear(2*atom_fea_len + nbr_fea_len, h_fea_len_dist)
        self.bond_distance_bn = nn.BatchNorm1d(h_fea_len)
        self.bond_distance_softplus = nn.Softplus()

        self.bond_constant = nn.Linear(2*atom_fea_len + nbr_fea_len, h_fea_len_const)
        self.bond_constant_bn = nn.BatchNorm1d(h_fea_len)    
        self.bond_const_softplus = nn.Softplus()

        
        
        if n_h_dist > 1:
            self.dist_fcs = nn.ModuleList([nn.Linear(h_fea_len_dist, h_fea_len_dist)
                                      for _ in range(n_h_dist-1)])
            self.dist_softpluses = nn.ModuleList([nn.Sigmoid() #LeakyReLU()
                                             for _ in range(n_h_dist-1)])
            self.dist_bn = nn.ModuleList([nn.BatchNorm1d(h_fea_len_dist)
                                             for _ in range(n_h_dist-1)])
            
        if n_h_const > 1:
            self.const_fcs = nn.ModuleList([nn.Linear(h_fea_len_const, h_fea_len_const)
                                      for _ in range(n_h_const-1)])
            self.const_softpluses = nn.ModuleList([nn.Sigmoid() #LeakyReLU()
                                             for _ in range(n_h_const-1)])
            self.const_bn = nn.ModuleList([nn.BatchNorm1d(h_fea_len_const)
                                             for _ in range(n_h_const-1)])
            
        self.bond_distance2 = nn.Linear(h_fea_len_dist, 1)
        self.bond_constant2 = nn.Linear(h_fea_len_const, 1)

        
        if self.energy_mode == "Morse" or self.energy_mode == "LJ":
            self.D_layer = nn.Linear(2*atom_fea_len + nbr_fea_len, h_fea_len_D)
            self.const_D_bn = nn.BatchNorm1d(h_fea_len_D)    
            self.D_softplus = nn.Softplus()
            self.D_sigmoid = nn.Sigmoid()
                
            if n_h_D > 1:
                self.D_fcs = nn.ModuleList([nn.Linear(h_fea_len_D, h_fea_len_D)
                                          for _ in range(n_h_D-1)])
                self.D_softpluses = nn.ModuleList([nn.Sigmoid() #LeakyReLU()
                                                 for _ in range(n_h_D-1)])
                self.D_bn = nn.ModuleList([nn.BatchNorm1d(h_fea_len_D)
                                                 for _ in range(n_h_D-1)])
            self.D_constant = nn.Linear(h_fea_len_D, 1)
            
                
        self.min_opt_steps=min_opt_steps
        self.max_opt_steps=max_opt_steps
        self.opt_step_size=opt_step_size


    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, nbr_fea_offset, crystal_atom_idx, atom_pos, nbr_pos, atom_pos_idx, cells, fixed_atom_mask, atom_pos_final):
#     def forward(self, atom_fea, nbr_fea_idx, nbr_fea_offset, crystal_atom_idx, atom_pos, cells, fixed_atom_mask, atom_pos_final):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        distances: torch.Tensor shape (N, 1)
          Storing connectivity information of atoms
        connection_atom_idx: torch.Tensor shape (N, 1)
          One hot encoding representation of the connectivity  
          
        Returns
        -------
        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution
        atom_fea_vis: nn.Variable shape (N, 1)  
          Contributions of atoms (distance <= 2)
          (All atoms (distance >2) are set to 0)
        atom_fea: nn.Variable shape (N, 1)  
          Per atom contributions

        """
        atom_pos = atom_pos.requires_grad_(True)
        atom_fea = self.embedding(atom_fea)
        
        free_atom_idx = np.where(fixed_atom_mask.cpu().numpy() == 0)[0]
        fixed_atom_idx = np.where(fixed_atom_mask.cpu().numpy() == 1)[0]

        distance = self.get_distance(atom_pos, cells, nbr_fea_offset, nbr_fea_idx)
#         nbr_fea = self.gdf.expand(distance)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        # Creating bond feature
        atom_features = atom_fea.unsqueeze(1).repeat(1, nbr_fea_idx.shape[1], 1)        
        nbr_features = atom_fea[nbr_fea_idx]
        
        bond_fea = torch.cat((atom_features, nbr_features, nbr_fea), dim=2)
        
        #First network to predict the adjustment to the initial distance for the bond springs

        bond_dist_fea = bond_fea
        #Set the bond spring distance to the correction plus the initial position
        
        bond_distance = self.bond_distance(bond_dist_fea)
        N, M, C = bond_distance.shape
        bond_distance = self.bond_distance_softplus(self.bond_distance_bn(bond_distance.view(-1, C)).view(N, M, C))

        if hasattr(self, 'dist_fcs') and hasattr(self, 'dist_softpluses'):
            for fc, softplus,bn in zip(self.dist_fcs, self.dist_softpluses, self.dist_bn):
                bond_distance = softplus(bn(fc(bond_distance).view(-1,C)).view(N,M,C))
        
        bond_distance = self.bond_distance_softplus((self.bond_distance2(bond_distance)+distance)) #+ distance
        
        #Second set of dense networks to predict the spring constant for each spring


        bond_const_fea = bond_fea
        bond_constant = self.bond_constant(bond_const_fea)
        N, M, C = bond_constant.shape
        bond_constant = self.bond_const_softplus(self.bond_constant_bn(bond_constant.view(-1, C)).view(N, M, C))
        
        if hasattr(self, 'const_fcs') and hasattr(self, 'const_softpluses'):
            for fc, softplus,bn in zip(self.const_fcs, self.const_softpluses, self.const_bn):
                bond_constant = softplus(bn(fc(bond_constant).view(-1,C)).view(N,M,C))
        
        bond_constant = self.bond_const_softplus(self.bond_constant2(bond_constant))
        
        
        if self.energy_mode == "Morse" or self.energy_mode == "LJ":
            const_D = bond_fea
            const_D = self.D_layer(const_D)
            N, M, C = const_D.shape
            const_D = self.D_softplus(self.const_D_bn(const_D.view(-1,C)).view(N,M,C))

            
            if hasattr(self, 'D_fcs') and hasattr(self, 'D_softpluses'):
                for fc, softplus,bn in zip(self.D_fcs, self.D_softpluses, self.D_bn):
                    const_D = softplus(bn(fc(const_D).view(-1,C)).view(N,M,C))
            if self.energy_mode == "Morse":
                const_D = self.D_softplus(self.D_constant(const_D))
            else:
                const_D = self.D_sigmoid(self.D_constant(const_D))

        
        steepest_descent_step=torch.FloatTensor([1.0])
        V = torch.tensor(0.0)
        save_track = [atom_pos]
        grad = torch.FloatTensor([100.0])
        step_count = 0
                                         
        while (torch.max(torch.abs(steepest_descent_step))>0.0005 and step_count<self.max_opt_steps) or step_count<self.min_opt_steps:

            distance = self.get_distance(atom_pos, cells, nbr_fea_offset, nbr_fea_idx)
            
            if self.energy_mode == "Morse":
                alpha = torch.sqrt(bond_constant/(2*const_D))
                potential_E = const_D * (1 - torch.exp(-alpha*(distance-bond_distance)))**2
            elif self.energy_mode == "LJ":
                bond_energy = bond_constant*(bond_distance-distance)**2
                LJ_energy = (const_D/len(const_D)) * ((bond_distance/distance)**12 - 2*(bond_distance/distance)**6)
                potential_E = LJ_energy + bond_energy
            else:
#                 bond_energy = torch.abs(bond_constant*(bond_distance-distance)**2.) 
                potential_E = bond_constant*(bond_distance-distance)**2
            
            grad_E = potential_E.sum() #.sum()
            grad = torch.autograd.grad(grad_E, atom_pos, retain_graph=True, create_graph=True)[0]
            
            grad[fixed_atom_idx] = 0
            if grad_E == torch.Tensor([float('inf')]).cuda():
                print('grad_E becomes inf')
            #detect if step is going off the rails
            if torch.max(torch.isnan(grad)) == 1:
                print('nan')
                return atom_pos[free_atom_idx]
            if torch.max(torch.abs(self.opt_step_size*grad))>5.0:
                print('blow up')
                return atom_pos[free_atom_idx]

            steepest_descent_step =  - self.opt_step_size*grad
    
            #### Momentum 
            V = self.momentum*V + (1-self.momentum)*grad
            atom_pos = atom_pos - self.opt_step_size * V
            step_count += 1

        return atom_pos[free_atom_idx], grad, atom_pos 

    def get_distance(self, atom_pos, cells, nbr_fea_offset, nbr_fea_idx):
        nbr_pos = atom_pos[nbr_fea_idx]
        differ = nbr_pos - atom_pos.unsqueeze(1)+ torch.bmm(nbr_fea_offset, cells)
        differ_sum = torch.sum(differ**2, dim=2)
    
        if self.energy_mode == "LJ":
            differ_sum = torch.clamp(differ_sum, min=1e-6)
        else:
            differ_sum = torch.clamp(differ_sum, min=1e-8)
        
        distance = torch.sqrt(differ_sum).unsqueeze(-1)            
        return distance
    
    def pooling(self, bond_fea_layer, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, 1)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        distances: torch.Tensor shape (N, 1)
          Storing connectivity information of atoms        
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            bond_fea_layer.data.shape[0]
#         assert sum([len(idx_map) for idx_map in crystal_fixed_atom_idx]) ==\
#             fixed_atom_idx.data.shape[0]
        
        
        summed_fea = torch.unsqueeze(torch.stack([torch.mean(bond_fea_layer[idx_map]) for idx_map in crystal_atom_idx]),1)
        
        
        return summed_fea
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step_size, var=None):

        assert dmin < dmax
        assert dmax - dmin > step_size
        self.filter = torch.arange(dmin, dmax+step_size, step_size).cuda()
        if var is None:
            var = step_size
        self.var = var

    def expand(self, distances):
        return torch.exp(-(distances - self.filter)**2 /
                      self.var**2)
    