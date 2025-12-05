from __future__ import print_function, division

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalContextModule(nn.Module):
    """
    Simple global pooling + gating module.

    For each crystal, we:
      - mean-pool atom features,
      - pass through a small gating MLP,
      - return a context vector per crystal.

    This context vector can then be broadcast back to all atoms in that crystal
    to provide a global signal (long-range interaction surrogate).
    """

    def __init__(self, atom_fea_len: int) -> None:
        super().__init__()
        self.gate_fc = nn.Linear(atom_fea_len, atom_fea_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, crystal_atom_fea: torch.Tensor, crystal_atom_idx: List[torch.LongTensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        crystal_atom_fea : Tensor, shape (N_atoms, atom_fea_len)
            Concatenated atom features for all crystals in the batch.
        crystal_atom_idx : list[LongTensor]
            Each element is a 1D LongTensor with the atom indices belonging
            to a given crystal.

        Returns
        -------
        context_vector : Tensor, shape (N_crystals, atom_fea_len)
        """
        context_list = []
        for atom_idx in crystal_atom_idx:
            c_fea = crystal_atom_fea[atom_idx]              # (n_i, atom_fea_len)
            c_vec = c_fea.mean(dim=0, keepdim=True)         # (1, atom_fea_len)
            context_list.append(c_vec)

        context_vector = torch.cat(context_list, dim=0)      # (N_crystals, atom_fea_len)
        context_vector = self.sigmoid(self.gate_fc(context_vector))
        return context_vector


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs with residual / skip connection.

    For each atom i, we aggregate information from its neighbors j:

        h_i^{new} = softplus( skip(h_i) + BN( Σ_j filter_ij * core_ij ) )

    where filter_ij and core_ij come from a gated MLP over
    [h_i, h_j, edge_ij].
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, use_skip: bool = True) -> None:
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.use_skip = use_skip

        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len,
                                 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

        if self.use_skip:
            self.skip_transform = nn.Linear(atom_fea_len, atom_fea_len)

    def forward(
        self,
        atom_in_fea: torch.Tensor,   # (N_atoms, atom_fea_len)
        nbr_fea: torch.Tensor,       # (N_atoms, max_num_nbr, nbr_fea_len)
        nbr_fea_idx: torch.Tensor,   # (N_atoms, max_num_nbr)
    ) -> torch.Tensor:
        N, M = nbr_fea_idx.shape

        # gather neighbor atom features: (N, M, atom_fea_len)
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        # central atom features expanded: (N, M, atom_fea_len)
        atom_self_fea = atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len)

        # concat [h_i, h_j, edge_ij]
        total_nbr_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=2)

        # gated MLP
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, 2 * self.atom_fea_len))
        total_gated_fea = total_gated_fea.view(N, M, 2 * self.atom_fea_len)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # aggregate over neighbors
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)   # (N, atom_fea_len)
        nbr_sumed = self.bn2(nbr_sumed)

        # residual / skip
        if self.use_skip:
            skip_val = self.skip_transform(atom_in_fea)
            out = self.softplus2(skip_val + nbr_sumed)
        else:
            out = self.softplus2(atom_in_fea + nbr_sumed)

        return out


class CrystalGraphConvNet(nn.Module):
    """
    Modified CGCNN that includes:
      - a learnable embedding from original atom features -> hidden size
      - multiple ConvLayer blocks with residual connections
      - optional GlobalContextModule, applied after each conv
      - mean pooling over atoms -> crystal representation
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        classification: bool = False,
        use_global_context: bool = True,
    ) -> None:
        super().__init__()

        self.classification = classification

        # Map raw atom features (including optional charge) to hidden dim
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Stacked conv layers
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len,
                          nbr_fea_len=nbr_fea_len,
                          use_skip=True)
                for _ in range(n_conv)
            ]
        )

        # Optional global context module
        self.use_global_context = use_global_context
        if use_global_context:
            self.global_context_module = GlobalContextModule(atom_fea_len)

        # After pooling
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        else:
            self.fcs = None
            self.softpluses = None

        # Output layer
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(
        self,
        atom_fea: torch.Tensor,             # (N_atoms, orig_atom_fea_len)
        nbr_fea: torch.Tensor,              # (N_atoms, max_num_nbr, nbr_fea_len)
        nbr_fea_idx: torch.Tensor,          # (N_atoms, max_num_nbr)
        crystal_atom_idx: List[torch.LongTensor],
    ) -> torch.Tensor:
        # Initial embedding
        atom_fea = self.embedding(atom_fea)

        # Convolutional blocks (+ optional global context)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

            if self.use_global_context:
                context_vec = self.global_context_module(atom_fea, crystal_atom_idx)
                # broadcast context to all atoms
                extended_context_list = []
                for i, idx_map in enumerate(crystal_atom_idx):
                    # context_vec[i]: (atom_fea_len,)
                    c_expanded = context_vec[i].unsqueeze(0).expand(len(idx_map), -1)
                    extended_context_list.append(c_expanded)
                extended_context = torch.cat(extended_context_list, dim=0)
                atom_fea = atom_fea + extended_context

        # Pooling over atoms for each crystal
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # Fully connected head
        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))
        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if self.fcs is not None and self.softpluses is not None:
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    @staticmethod
    def pooling(atom_fea: torch.Tensor, crystal_atom_idx: List[torch.LongTensor]) -> torch.Tensor:
        """
        Mean pooling over atoms for each crystal.

        Parameters
        ----------
        atom_fea : Tensor, shape (N_atoms, atom_fea_len)
        crystal_atom_idx : list[LongTensor]
            Each element is a vector of atom indices belonging to one crystal.

        Returns
        -------
        pooled_fea : Tensor, shape (N_crystals, atom_fea_len)
        """
        assert sum(len(idx_map) for idx_map in crystal_atom_idx) == atom_fea.shape[0]
        pooled_fea = []
        for idx_map in crystal_atom_idx:
            fea = atom_fea[idx_map].mean(dim=0, keepdim=True)
            pooled_fea.append(fea)
        return torch.cat(pooled_fea, dim=0)