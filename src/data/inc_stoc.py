from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ..physics.inc_stoc_vort import simulate

import pytorch_lightning as L
from torch.utils.data import Subset


class IncStocDataset(Dataset):
    def __init__(
        self,
        path: str,
        ctx_len: int,
        pred_len: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.arr = np.load(path, allow_pickle=True)
        self.vorticity = self.arr["vorticity"].astype(np.float32)

        self.T = self.vorticity.shape[1]
        self.C = self.vorticity.shape[2]
        self.H = self.vorticity.shape[3]
        self.W = self.vorticity.shape[4]

        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.stride = stride

        self.index = []
        num_trajectories = self.vorticity.shape[0]
        for traj_idx in range(num_trajectories):
            max_start = self.T - (self.ctx_len + self.pred_len)
            for start_t in range(0, max_start + 1, self.stride):
                self.index.append((traj_idx, start_t))

    def __len__(self) -> int:
        return len(self.index)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        # TODO
        return x

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m, s = self.index[i]
        e_ctx = s + self.ctx_len
        e_all = e_ctx + self.pred_len
        seq = self.vorticity[m, s:e_all]
        x = seq[: self.ctx_len]
        y = seq[self.ctx_len:]
        x = self._norm(x)
        y = self._norm(y)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def load_sequence(self, start_frame: int, len: int, sim: int = None) -> torch.Tensor:
        traj = simulate(warmup_seed=sim if sim is not None else 1, T=len, warmup=start_frame)
        return torch.tensor(traj).float()

    def load_sequences(self, start_frame: int, length: int, num_seq: int) -> List[torch.Tensor]:
        sequences = []
        for i in range(num_seq):
            sequence = self.load_sequence(start_frame=start_frame, len=length, sim=i)
            sequences.append(sequence)
        return sequences

    def load_multiple_sequences(self, num_seq: int, start_frame: int, len: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs_seeds = [i + 100 for i in range(num_seq)]

        warmup, seqs_list = simulate(
            warmup_seed=seed, trajectory_seeds=seqs_seeds, T=len, warmup=start_frame
        )
        seqs = np.stack(seqs_list, axis=0)

        return torch.tensor(warmup).float(), torch.tensor(seqs).float()


class IncStocDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        ctx_len: int,
        pred_len: int,
        stride: int,
        sim_fields: List[str],
        sim_params: List[str],
        num_workers: int,
        **_: Any
    ) -> None:

        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.stride = stride
        self.sim_fields = sim_fields
        self.sim_params = sim_params
        self.num_workers = num_workers

    def setup(self, stage: str = None) -> None:
        self.train_set = IncStocDataset(
            path=self.data_path,
            ctx_len=self.ctx_len,
            pred_len=self.pred_len,
            stride=self.stride,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.train_set, [0]),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.train_set, [0]),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
