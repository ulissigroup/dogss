from __future__ import print_function, division

import torch
import pickle

import torch.nn as nn
import numpy as np
from ase.geometry import wrap_positions
import copy

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs using Message Passing
    """
    def __init__(self, node_fea_size, edge_fea_size):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        node_fea_size: int
          Number of node hidden features.
        edge_fea_size: int
          Number of edge features.
        """
        super(ConvLayer, self).__init__()
        self.node_fea_size = node_fea_size
        self.edge_fea_size = edge_fea_size
        
        self.fc_pre_node = nn.Linear(2*self.node_fea_size+self.edge_fea_size,
                                 2*self.node_fea_size)

        self.node_bn1 = nn.BatchNorm1d(2*self.node_fea_size)
        self.node_bn2 = nn.BatchNorm1d(self.node_fea_size)
        
        self.fc_pre_edge = nn.Linear(2*self.node_fea_size+self.edge_fea_size,
                                 2*self.edge_fea_size)
        
        self.edge_bn1 = nn.BatchNorm1d(2*self.edge_fea_size)
        self.edge_bn2 = nn.BatchNorm1d(self.edge_fea_size)
        
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.LeakyReLU()
        
    def forward(self, node_fea_in, edge_fea, edge_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        node_fea_in: Variable(torch.Tensor) shape (N, node_fea_size)
          Atom hidden features before convolution
        edge_fea: Variable(torch.Tensor) shape (N, M, edge_fea_size)
          Bond features of each atom's M neighbors
        edge_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        updated_node: nn.Variable shape (N, node_fea_size)
          Updated node features
        updated_edge: nn.Variable shape (N, M, edge_fea_size)
          Updated edge features
        """
        N, M = edge_idx.shape
        bond_fea = node_fea_in[edge_idx, :]
        
        z = torch.cat(
            [node_fea_in.unsqueeze(1).expand(N, M, self.node_fea_size),
             bond_fea, edge_fea], dim=2)
        
        ## Node Update
        z_node = self.fc_pre_node(z)
        z_node = self.node_bn1(z_node.view(
            -1, self.node_fea_size*2)).view(N, M, self.node_fea_size*2)
        
        node_filter, node_core = z_node.chunk(2, dim=2)
        node_filter = self.sigmoid(node_filter)
        node_core = self.softplus(node_core)
        
        msg_node = node_filter * node_core
        aggr_msg_node = torch.sum(msg_node, dim=1)
        aggr_msg_node = self.node_bn2(aggr_msg_node)
        updated_node = self.softplus(node_fea_in + aggr_msg_node)
        
        ## Edge Update
        z_edge = self.fc_pre_edge(z)
        z_edge = self.edge_bn1(z_edge.view(
            -1, self.edge_fea_size*2)).view(N, M, self.edge_fea_size*2)
        
        edge_filter, edge_core = z_edge.chunk(2, dim=2)
        edge_filter = self.sigmoid(edge_filter)
        edge_core = self.softplus(edge_core)
        msg_edge = edge_filter * edge_core
        msg_edge = self.edge_bn2(msg_edge.view(-1,self.edge_fea_size)).view(N,M, self.edge_fea_size)
        updated_edge = self.softplus(edge_fea + msg_edge)
        
        return updated_node, updated_edge
        
class DOGSS(nn.Module):

    def __init__(self, 
                 orig_node_fea_size, 
                 edge_fea_size,
                 node_fea_size=64, 
                 n_conv=3, 
                 h_fea_len=128, 
                 h_fea_len_dist=128, 
                 h_fea_len_const=128, 
                 h_fea_len_D=128, 
                 n_h_dist=1, 
                 n_h_const=1, 
                 n_h_D = 1,
                 momentum=0.9, 
                 max_opt_steps=300, 
                 min_opt_steps=10, 
                 opt_step_size=0.3, 
                 energy_mode="Harmonic",
                ):
        
        """
        Initialize DOGSS.
        """
        super(DOGSS, self).__init__()
        
        self.energy_mode = energy_mode
        self.momentum = momentum
        self.embedding = nn.Linear(orig_node_fea_size, node_fea_size)
        
        self.convs = nn.ModuleList(
            [
                ConvLayer(node_fea_size=node_fea_size,
                                    edge_fea_size=edge_fea_size)
                                    for _ in range(n_conv)
            ]
        )
        
        self.conv_to_bond_distance = nn.Linear(2*node_fea_size + edge_fea_size, h_fea_len_dist)
        self.bond_distance_bn = nn.BatchNorm1d(h_fea_len_dist)

        self.conv_to_bond_constant = nn.Linear(2*node_fea_size + edge_fea_size, h_fea_len_const)
        self.bond_constant_bn = nn.BatchNorm1d(h_fea_len_const)    
        
        self.softplus = nn.Softplus()        
        self.sigmoid = nn.Sigmoid()
        
        if n_h_dist > 1:
            self.dist_fcs = nn.ModuleList([nn.Linear(h_fea_len_dist, h_fea_len_dist)
                                      for _ in range(n_h_dist-1)])
            self.dist_softpluses = nn.ModuleList([nn.Sigmoid()
                                             for _ in range(n_h_dist-1)])
            self.dist_bn = nn.ModuleList([nn.BatchNorm1d(h_fea_len_dist)
                                             for _ in range(n_h_dist-1)])
            
        if n_h_const > 1:
            self.const_fcs = nn.ModuleList([nn.Linear(h_fea_len_const, h_fea_len_const)
                                      for _ in range(n_h_const-1)])
            self.const_softpluses = nn.ModuleList([nn.Sigmoid() 
                                             for _ in range(n_h_const-1)])
            self.const_bn = nn.ModuleList([nn.BatchNorm1d(h_fea_len_const)
                                             for _ in range(n_h_const-1)])
            
        self.bond_distance = nn.Linear(h_fea_len_dist, 1)
        self.bond_constant = nn.Linear(h_fea_len_const, 1)

        if self.energy_mode == "Morse" or self.energy_mode == "LJ":
            self.D_layer = nn.Linear(2*node_fea_size + edge_fea_size, h_fea_len_D)
            self.const_D_bn = nn.BatchNorm1d(h_fea_len_D)    
                
            if n_h_D > 1:
                self.D_fcs = nn.ModuleList([nn.Linear(h_fea_len_D, h_fea_len_D)
                                          for _ in range(n_h_D-1)])
                self.D_softpluses = nn.ModuleList([nn.Sigmoid() 
                                                 for _ in range(n_h_D-1)])
                self.D_bn = nn.ModuleList([nn.BatchNorm1d(h_fea_len_D)
                                                 for _ in range(n_h_D-1)])
            self.D_constant = nn.Linear(h_fea_len_D, 1)
                
        self.min_opt_steps=min_opt_steps
        self.max_opt_steps=max_opt_steps
        self.opt_step_size=opt_step_size
                
    def forward(self, node_fea, edge_fea, edge_idx, nbr_offset, atom_pos, cells, ads_tag_base, fixed_atom_mask, atom_pos_final):

        """
        Forward pass

        Parameters
        ----------

    node_fea: torch.Tensor shape (N, orig_node_fea_size)
      Atom features from atom type
    edge_fea: torch.Tensor shape (N, M, edge_fea_size)
      Bond features of each atom's M neighbors
    edge_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    nbr_offset: torch.Tensor shape (N, M, 3)
      Indices of the unit cell where each atom is located 
    atom_pos: torch.LongTensor shape (N, 3)
      Position of each atom in the initial structures
    cells: torch.LongTensor shape (N, 3, 3)
      Unit cell vectors
    ads_tag_base: torch.LongTensor shape (N, 1)
      Tagging adsorbate atoms
    fixed_atom_mask: torch.LongTensor shape (N, 1)
      Tagging fixed atoms
    atom_pos_final: torch.LongTensor shape (N, 3)
      Position of each atom in the final/relaxed structures 
          
        Returns
        -------
        prediction: nn.Variable shape (N[free_atoms_idx], 3)
          Position of each of the free atoms in the predicted/relaxed structures

        """
        
        atom_pos = atom_pos.requires_grad_(True)
        free_atom_idx = np.where(fixed_atom_mask.cpu().numpy() == 0)[0]
        fixed_atom_idx = np.where(fixed_atom_mask.cpu().numpy() == 1)[0]
        ads_idx = np.where(ads_tag_base.cpu().numpy() == 1)[0]
        
        distance = self.get_distance(atom_pos, cells, nbr_offset, edge_idx)
        
        node_fea = self.embedding(node_fea)
        for conv_func in self.convs:
            node_fea, edge_fea = conv_func(node_fea, edge_fea, edge_idx)

        # Creating bond feature
        atom_features = node_fea.unsqueeze(1).repeat(1, edge_idx.shape[1], 1)        
        nbr_features = node_fea[edge_idx]
        
        bond_fea = torch.cat((atom_features, nbr_features, edge_fea), dim=2)
        

        bond_dist_fea = bond_fea
        ## Set the bond spring distance to the correction plus the initial position
        bond_distance = self.conv_to_bond_distance(bond_dist_fea)
        N, M, C = bond_distance.shape
        
        if hasattr(self, 'dist_fcs') and hasattr(self, 'dist_softpluses'):
            bond_distance = self.softplus(self.bond_distance_bn(bond_distance.view(-1, C)).view(N, M, C))
            for fc, softplus,bn in zip(self.dist_fcs, self.dist_softpluses, self.dist_bn):
                bond_distance = softplus(bn(fc(bond_distance).view(-1,C)).view(N,M,C))
            bond_distance = self.softplus(self.bond_distance(bond_distance)+ distance) #+ distance
            
        else:
            bond_distance = self.softplus(self.bond_distance_bn(bond_distance.view(-1, C)).view(N, M, C) + distance)
            bond_distance = torch.mean(bond_distance, dim=2).unsqueeze(-1)       
            
        ## Second set of dense networks to predict the spring constant for each spring
        bond_const_fea = bond_fea
        bond_constant = self.conv_to_bond_constant(bond_const_fea)
        N, M, C = bond_constant.shape
        
        if hasattr(self, 'const_fcs') and hasattr(self, 'const_softpluses'):
            bond_constant = self.softplus(self.bond_constant_bn(bond_constant.view(-1, C)).view(N, M, C))
            for fc, softplus,bn in zip(self.const_fcs, self.const_softpluses, self.const_bn):
                bond_constant = softplus(bn(fc(bond_constant).view(-1,C)).view(N,M,C))
            bond_constant = self.softplus(self.bond_constant(bond_constant)) / len(bond_constant)
        else:
            bond_constant = self.softplus(self.bond_constant_bn(bond_constant.view(-1, C)).view(N, M, C))
            bond_constant = torch.mean(bond_constant, dim=2).unsqueeze(-1) / len(bond_constant)
        
        ## If potential mode is either Morse or LJ, it requires one additional parameter (D)
        if self.energy_mode == "Morse" or self.energy_mode == "LJ":
            const_D = bond_fea
            const_D = self.D_layer(const_D)
            N, M, C = const_D.shape
            const_D = self.softplus(self.const_D_bn(const_D.view(-1,C)).view(N,M,C))
            
            if hasattr(self, 'D_fcs') and hasattr(self, 'D_softpluses'):
                for fc, softplus,bn in zip(self.D_fcs, self.D_softpluses, self.D_bn):
                    const_D = softplus(bn(fc(const_D).view(-1,C)).view(N,M,C))
                const_D = self.softplus(self.D_constant(const_D))
            else:
                const_D = self.softplus(torch.mean(const_D, dim=2).unsqueeze(-1))
            
            if self.energy_mode == "Morse":
                const_D = self.softplus(self.D_constant(const_D))
            else:
                const_D = self.sigmoid(self.D_constant(const_D))

        ## Differentiable Optimization Loop (Gradient Descent)
        steepest_descent_step=torch.FloatTensor([1.0])
        V = torch.tensor(0.0)
        save_track = [atom_pos]
        grad = torch.FloatTensor([100.0])
        step_count = 0
                                         
        while (torch.max(torch.abs(steepest_descent_step))>0.0005 and step_count<self.max_opt_steps) or step_count<self.min_opt_steps:

            distance = self.get_distance(atom_pos, cells, nbr_offset, edge_idx)
            
            if self.energy_mode == "Morse":
                alpha = torch.sqrt(torch.abs(bond_constant)/torch.abs(2*const_D))
                potential_E = const_D * (1 - torch.exp(-alpha*(distance-bond_distance)))**2
            elif self.energy_mode == "LJ":
                LJ_energy = bond_constant * ((bond_distance/distance)**12 - 2*(bond_distance/distance)**6)
                potential_E = LJ_energy
            else:
                potential_E = bond_constant*(bond_distance-distance)**2
            
            grad_E = potential_E.sum()

            grad = torch.autograd.grad(grad_E, atom_pos, retain_graph=True, create_graph=True)[0]
            
            grad[fixed_atom_idx] = 0
            if grad_E == torch.Tensor([float('inf')]).cuda():
                print('grad_E becomes inf')
            # detect if step is going off the rails
            if torch.max(torch.isnan(grad)) == 1:
                print('nan')
                return atom_pos[free_atom_idx]
            if torch.max(torch.abs(self.opt_step_size*grad))>5.0:
                print('blow up')
                return atom_pos[free_atom_idx]

            steepest_descent_step =  - self.opt_step_size*grad
    
            ## Momentum 
            V = self.momentum*V + (1-self.momentum)*grad
            atom_pos = atom_pos - self.opt_step_size * V
            step_count += 1

        return atom_pos[free_atom_idx]
  
    def get_distance(self, atom_pos, cells, nbr_offset, edge_idx):
        nbr_pos = atom_pos[edge_idx]
        differ = nbr_pos - atom_pos.unsqueeze(1)+ torch.bmm(nbr_offset, cells)
        differ_sum = torch.sum(differ**2, dim=2)
    
        if self.energy_mode == "LJ":
            differ_sum = torch.clamp(differ_sum, min=1e-6)
        else:
            differ_sum = torch.clamp(differ_sum, min=1e-8)
        
        distance = torch.sqrt(differ_sum).unsqueeze(-1)            
        return distance
