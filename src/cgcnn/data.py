import csv
import functools
import json
import os
import random
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# ----------------------------------------------------------------------------- #
# Atom feature initialization
# ----------------------------------------------------------------------------- #

class AtomCustomJSONInitializer:
    """
    Load atom-wise feature vectors from a JSON file.

    JSON format:
        {
            "1": [feat_1, feat_2, ...],
            "6": [...],
            ...
        }
    where keys are atomic numbers (as strings) and values are float lists.
    """

    def __init__(self, json_path: str) -> None:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Atom feature JSON not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        self._embedding: Dict[int, np.ndarray] = {}
        for key, value in data.items():
            Z = int(key)
            self._embedding[Z] = np.array(value, dtype=float)

        if not self._embedding:
            raise ValueError(f"No atom features found in {json_path}.")

        self._feat_len = len(next(iter(self._embedding.values())))

    def get_atom_fea(self, atomic_number: int) -> np.ndarray:
        if atomic_number not in self._embedding:
            raise KeyError(f"No features found for Z={atomic_number}.")
        return self._embedding[atomic_number]

    @property
    def feature_length(self) -> int:
        return self._feat_len


class GaussianDistance:
    """
    Filter to expand distances using Gaussian basis.

    For each scalar distance d, returns:
        [exp(-(d - mu_i)^2 / var^2) for mu_i in filter_centers]
    """

    def __init__(self, dmin: float, dmax: float, step: float, var: Optional[float] = None) -> None:
        assert dmax > dmin
        assert step > 0
        self.filter_centers = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        distances : np.ndarray
            Distance array of shape (...,).

        Returns
        -------
        expanded : np.ndarray
            Expanded distances of shape (..., n_centers).
        """
        d = np.expand_dims(distances, axis=-1)  # (..., 1)
        return np.exp(-(d - self.filter_centers) ** 2 / self.var ** 2)


# ----------------------------------------------------------------------------- #
# CIF dataset
# ----------------------------------------------------------------------------- #

class CIFData(Dataset):
    """
    Dataset for MOF/solid-state structures stored as CIF + id_prop.csv.

    root_dir layout:
        root_dir/
          id_prop.csv
          atom_init.json
          atom_init_<embedding>.json  (optional)
          <cif_id_1>.cif
          <cif_id_2>.cif
          ...

    id_prop.csv format:
        cif_id,target
        QMOF-0001,1.23
        QMOF-0002,2.34
        ...

    If `partial_charge_file` is provided, it should be a JSON mapping:
        {
          "QMOF-0001": [q1, q2, ..., qN],
          ...
        }
    """

    def __init__(
        self,
        root_dir: str,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        dmin: float = 0.0,
        step: float = 0.2,
        random_seed: int = 123,
        embedding_name: Optional[str] = None,
        partial_charge_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

        id_prop_file = os.path.join(self.root_dir, "id_prop.csv")
        if not os.path.exists(id_prop_file):
            raise FileNotFoundError(f"id_prop.csv does not exist in {root_dir}")

        with open(id_prop_file, "r") as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Optional header handling: if first row is header, skip it
        if self.id_prop_data and not self._is_float(self.id_prop_data[0][1]):
            self.id_prop_data = self.id_prop_data[1:]

        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        # Choose atom_init file
        if embedding_name is None or embedding_name == "default":
            atom_init_file = os.path.join(self.root_dir, "atom_init.json")
            print(f"Using default atom_init.json: {atom_init_file}")
        else:
            atom_init_file = os.path.join(self.root_dir, f"atom_init_{embedding_name}.json")
            print(f"Using atom_init_{embedding_name}.json: {atom_init_file}")

        if not os.path.exists(atom_init_file):
            raise FileNotFoundError(f"{atom_init_file} does not exist!")

        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        # Partial charges
        if partial_charge_file is not None:
            if not os.path.exists(partial_charge_file):
                raise FileNotFoundError(f"{partial_charge_file} does not exist!")
            with open(partial_charge_file, "r") as f:
                self.partial_charges: Dict[str, List[float]] = json.load(f)
        else:
            self.partial_charges = None

    def __len__(self) -> int:
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx: int):
        cif_id, target_str = self.id_prop_data[idx]
        target = float(target_str)

        cif_path = os.path.join(self.root_dir, cif_id + ".cif")
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

        crystal = Structure.from_file(cif_path)

        # Atom features from JSON initializer
        atom_fea = np.vstack(
            [self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
        )  # (n_atoms, feat_len)

        # Append partial charge as extra column if available
        if self.partial_charges is not None:
            charges_list = self.partial_charges.get(cif_id, [])
            charges = np.array(charges_list, dtype=float)
            if charges.size != atom_fea.shape[0]:
                warnings.warn(
                    f"Number of charges for {cif_id} ({charges.size}) "
                    f"does not match number of atoms ({atom_fea.shape[0]}). "
                    f"Skipping charges for this structure."
                )
            else:
                charges = charges.reshape(-1, 1)
                atom_fea = np.concatenate([atom_fea, charges], axis=1)

        atom_fea = torch.tensor(atom_fea, dtype=torch.float)

        # Neighbor information
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for i, nbr in enumerate(all_nbrs):
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_id} has fewer than max_num_nbr neighbors for atom {i}. "
                    f"Consider increasing radius."
                )
                # pad with dummy neighbors
                nbr_fea_idx.append(
                    [nbr_j[2] for nbr_j in nbr] + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    [nbr_j[1] for nbr_j in nbr]
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append([nbr_j[2] for nbr_j in nbr[: self.max_num_nbr]])
                nbr_fea.append([nbr_j[1] for nbr_j in nbr[: self.max_num_nbr]])

        nbr_fea_idx = np.array(nbr_fea_idx, dtype=int)            # (n_atoms, max_num_nbr)
        nbr_fea = np.array(nbr_fea, dtype=float)                  # (n_atoms, max_num_nbr)
        nbr_fea = self.gdf.expand(nbr_fea)                        # (n_atoms, max_num_nbr, n_gauss)

        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float)
        nbr_fea_idx = torch.tensor(nbr_fea_idx, dtype=torch.long)
        target = torch.tensor([target], dtype=torch.float)

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False


# ----------------------------------------------------------------------------- #
# Collate and data loaders
# ----------------------------------------------------------------------------- #

def collate_pool(dataset_list: List[Tuple]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.LongTensor]], torch.Tensor, List[str]]:
    """
    Collate function to combine multiple CIFData samples into a batch.

    Input:
        dataset_list: list of ( (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id )

    Returns:
        input_batch: (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        target:      tensor of shape (batch_size, 1)
        cif_ids:     list of cif_id strings
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_target = []
    batch_cif_ids = []
    crystal_atom_idx: List[torch.LongTensor] = []

    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id in dataset_list:
        n_i = atom_fea.shape[0]

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)

        idx_map = torch.arange(base_idx, base_idx + n_i, dtype=torch.long)
        crystal_atom_idx.append(idx_map)

        batch_target.append(target)
        batch_cif_ids.append(cif_id)

        base_idx += n_i

    batch_atom_fea = torch.cat(batch_atom_fea, dim=0)
    batch_nbr_fea = torch.cat(batch_nbr_fea, dim=0)
    batch_nbr_fea_idx = torch.cat(batch_nbr_fea_idx, dim=0)
    batch_target = torch.stack(batch_target, dim=0)

    return (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx), batch_target, batch_cif_ids


def get_train_val_test_loader(
    dataset: Dataset,
    collate_fn,
    batch_size: int,
    train_ratio: Optional[float] = None,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    pin_memory: bool = False,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    return_test: bool = True,
):
    """
    Create DataLoader objects for train/val/(test) splits.

    Either use *_ratio or *_size. If sizes are given, they take precedence.
    """
    total_size = len(dataset)
    indices = list(range(total_size))

    # Training size
    if train_size is not None:
        train_end = min(train_size, total_size)
    elif train_ratio is not None:
        train_end = int(total_size * train_ratio)
    else:
        # default: 1 - val_ratio - test_ratio
        train_end = int(total_size * (1.0 - val_ratio - test_ratio))

    # Remaining for val + test
    remaining = total_size - train_end
    if remaining < 0:
        raise ValueError("Train/val/test split is inconsistent with dataset size.")

    # Validation size
    if val_size is not None:
        val_end = train_end + min(val_size, remaining)
    else:
        # proportion of remaining
        val_end = train_end + int(remaining * (val_ratio / (val_ratio + test_ratio + 1e-8)))

    # Test size
    remaining_after_val = total_size - val_end
    if test_size is not None:
        test_end = val_end + min(test_size, remaining_after_val)
    else:
        test_end = total_size

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:test_end]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader