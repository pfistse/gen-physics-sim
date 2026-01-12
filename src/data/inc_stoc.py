from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ..solver.inc_stoc_vort_jax import simulate
import rootutils

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
        with np.load(path, allow_pickle=True) as _arr:
            self.vorticity = _arr["traj"].astype(np.float32).copy()
        self._cache_dir = os.path.join(os.path.dirname(path), "cached")

        self.T = self.vorticity.shape[1]
        self.C = self.vorticity.shape[2]
        self.H = self.vorticity.shape[3]
        self.W = self.vorticity.shape[4]
        assert self.H == self.W, "expected square fields"
        self.downsample = int(self.W)

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

    def _cache_file(self, start_frame: int, length: int, batch: int, seed: int, same_warmup: bool) -> str:
        suffix = (
            f"warmup{start_frame}_len{length}_B{batch}_seed{seed}_down{self.downsample}_"
            f"samewarmup{int(same_warmup)}.npz"
        )
        return os.path.join(self._cache_dir, suffix)

    def generate_sequence(self, start_frame: int, len: int, sim: int = None, seed: int = 42) -> torch.Tensor:
        """Generate or load one simulator sequence.

        result: [len, C, H, W]
        """
        os.makedirs(self._cache_dir, exist_ok=True)
        same_warmup = True
        cache_path = self._cache_file(
            start_frame=start_frame,
            length=len,
            batch=1,
            seed=seed,
            same_warmup=same_warmup,
        )
        if os.path.exists(cache_path):
            with np.load(cache_path) as arr:
                return torch.from_numpy(arr["traj"][0].copy()).float()

        warmup, traj = simulate(
            seed=seed,
            B=1,
            T=len,
            warmup=start_frame,
            downsample_to=self.downsample,
            same_warmup=same_warmup,
        )
        np.savez_compressed(cache_path, warmup=warmup, traj=traj)

        return torch.from_numpy(traj[0]).float()

    def generate_sequences(
        self,
        num_seq: int,
        start_frame: int,
        len: int,
        seed: int = 42,
        same_warmup: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate or load simulator sequences.

        warmup: [max(1, num_seq), T_warmup, C, H, W]
        traj: [num_seq, len, C, H, W]
        """
        os.makedirs(self._cache_dir, exist_ok=True)
        cache_path = self._cache_file(
            start_frame=start_frame,
            length=len,
            batch=num_seq,
            seed=seed,
            same_warmup=same_warmup,
        )
        if os.path.exists(cache_path):
            with np.load(cache_path) as arr:
                warmup = arr["warmup"].copy()
                traj = arr["traj"].copy()
            return torch.from_numpy(warmup).float(), torch.from_numpy(traj).float()
        warmup, traj = simulate(
            seed=seed,
            B=num_seq,
            T=len,
            warmup=start_frame,
            same_warmup=same_warmup,
            downsample_to=self.downsample,
        )
        np.savez_compressed(cache_path, warmup=warmup, traj=traj)
        return torch.from_numpy(warmup).float(), torch.from_numpy(traj).float()

    def _list_cached_sequence_files(
        self,
        start_frame: int,
        length: int,
        batch: int,
        same_warmup: bool,
    ) -> List[Path]:
        os.makedirs(self._cache_dir, exist_ok=True)
        pattern = (
            f"warmup{start_frame}_len{length}_B{batch}_seed*_down{self.downsample}_"
            f"samewarmup{int(same_warmup)}.npz"
        )
        return sorted(Path(self._cache_dir).glob(pattern))

    @staticmethod
    def _seed_from_cache_name(name: str) -> int:
        for token in name.split("_"):
            if token.startswith("seed"):
                return int(token.replace("seed", ""))
        raise ValueError(f"Cannot parse seed from cache file {name}")

    @staticmethod
    def _downsample_from_cache_name(name: str) -> int:
        for token in name.split("_"):
            if token.startswith("down"):
                return int(token.replace("down", ""))
        raise ValueError(f"Cannot parse downsample from cache file {name}")

    def load_cached_sequences(
        self,
        start_frame: int,
        len: int,
        batch: int,
        seeds: Optional[List[int]] = None,
        same_warmup: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        files = self._list_cached_sequence_files(
            start_frame=start_frame,
            length=len,
            batch=batch,
            same_warmup=same_warmup,
        )
        if not files:
            return []

        seed_to_file: Dict[int, Path] = {}
        for path in files:
            seed = self._seed_from_cache_name(path.name)
            seed_to_file[seed] = path

        if seeds is not None:
            missing = [seed for seed in seeds if seed not in seed_to_file]
            assert (
                len(missing) == 0
            ), f"cached sequences missing for seeds {missing}"
            selected = [(seed, seed_to_file[seed]) for seed in seeds]
        else:
            selected = sorted(seed_to_file.items())

        batches: List[Dict[str, torch.Tensor]] = []
        for seed, path in selected:
            with np.load(path, allow_pickle=False) as arr:
                warmup = torch.from_numpy(arr["warmup"].copy()).float()
                traj = torch.from_numpy(arr["traj"].copy()).float()
            batches.append({"seed": seed, "warmup": warmup, "traj": traj})
        return batches

    def load_sequence(self, len: int, sim: int = 0) -> torch.Tensor:
        """Return a sequence window from t=0.

        result: [len, C, H, W]
        """
        assert 0 <= sim < self.vorticity.shape[0], "sim index out of range"
        assert len <= self.T, "frame window out of range"
        return torch.from_numpy(self.vorticity[sim, :len]).float()

    def load_sequences(self, len: int, num_seq: int) -> List[torch.Tensor]:
        """Return multiple windows from t=0.

        result: list[[len, C, H, W]]
        """
        B = min(int(num_seq), self.vorticity.shape[0])
        return [self.load_sequence(len=len, sim=i) for i in range(B)]


class IncStocCachedDataset(Dataset):
    def __init__(self, entries: List[Tuple[int, Path]]) -> None:
        super().__init__()
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seed, path = self.entries[index]
        with np.load(path, allow_pickle=False) as arr:
            warmup = torch.from_numpy(arr["warmup"].copy()).float()
            traj = torch.from_numpy(arr["traj"].copy()).float()
        return {
            "warmup": warmup,
            "traj": traj,
            "seed": int(seed),
        }


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
        num_channels: int,
        num_workers: int,
        test_seq_seeds: Optional[List[int]] = None,
        **_: Any
    ) -> None:

        super().__init__()

        # Keep raw path; resolve on each worker in setup()
        self.data_path = data_path
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.stride = stride
        self.sim_fields = sim_fields
        self.sim_params = sim_params
        self.num_channels = num_channels
        self.num_workers = num_workers
        self.test_seq_seeds = list(test_seq_seeds) if test_seq_seeds is not None else None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.cached_eval_set = None
        self.sim_dataset = None

    def setup(self, stage: str = None) -> None:
        # Resolve path here so each worker maps it to its local repo root
        p = os.path.expandvars(os.path.expanduser(self.data_path))
        if not os.path.isabs(p):
            root = rootutils.find_root(search_from=__file__, indicator=".project-root")
            p = str(Path(root) / p)
        self.train_set = IncStocDataset(
            path=p,
            ctx_len=self.ctx_len,
            pred_len=self.pred_len,
            stride=self.stride,
        )
        self.sim_dataset = self.train_set

        cached_entries = self._build_cached_eval_entries()
        if cached_entries:
            self.cached_eval_set = IncStocCachedDataset(cached_entries)
            self.val_set = self.cached_eval_set
            self.test_set = self.cached_eval_set
        else:
            self.cached_eval_set = None
            self.val_set = Subset(self.train_set, [0])
            self.test_set = self.train_set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True  # TODO check
        )

    def val_dataloader(self) -> DataLoader:
        if self.cached_eval_set is not None:
            return DataLoader(
                self.cached_eval_set,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self._cached_collate,
            )
        return DataLoader(
            Subset(self.train_set, [0]),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.cached_eval_set is not None:
            return DataLoader(
                self.cached_eval_set,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self._cached_collate,
            )
        return DataLoader(
            Subset(self.train_set, [0]),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _cached_collate(batch):
        assert len(batch) == 1, "cached evaluation uses batch_size=1"
        return batch[0]

    def _build_cached_eval_entries(self) -> List[Tuple[int, Path]]:
        if self.train_set is None:
            return []

        cache_dir = Path(self.train_set._cache_dir)
        if not cache_dir.exists():
            return []

        allowed = set(self.test_seq_seeds) if self.test_seq_seeds else None
        entries: List[Tuple[int, Path]] = []
        for path in sorted(cache_dir.glob("*.npz")):
            name = path.name
            try:
                seed = self.train_set._seed_from_cache_name(name)
                down = self.train_set._downsample_from_cache_name(name)
            except ValueError:
                continue

            if down != self.train_set.downsample:
                continue
            if allowed and seed not in allowed:
                continue
            entries.append((seed, path))
        return entries
