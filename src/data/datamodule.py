import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


FIELD_FILE_MAP = {
    "vel": "velocity",
    "pres": "pressure",
}


def _load_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        # dummy generator stores under 'data'
        if "data" in data:
            return data["data"]
        # fallback for numpy default key
        return data[list(data.files)[0]]
    else:
        return data


class PhysicsDataset(Dataset):
    """Dataset loading sequences of physics simulation frames."""

    def __init__(
        self,
        root: str,
        sim_folders: List[str],
        frame_range: Tuple[int, int],
        conditioning_length: int,
        stride: int,
        sim_fields: List[str],
        sim_params: List[str],
        field_mean: Dict[str, List[float]],
        field_std: Dict[str, List[float]],
        param_mean: Dict[str, List[float]],
        param_std: Dict[str, List[float]],
        normalize: bool = True,
        target_length: int = 1,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.sim_folders = sim_folders
        self.frame_range = frame_range
        self.conditioning_length = conditioning_length
        self.stride = stride
        self.sim_fields = sim_fields
        self.sim_params = sim_params
        self.target_length = target_length
        self.field_mean = {
            k: np.asarray(v, dtype=np.float32) for k, v in field_mean.items()
        }
        self.field_std = {
            k: np.asarray(v, dtype=np.float32) for k, v in field_std.items()
        }
        self.param_mean = {
            k: np.asarray(v, dtype=np.float32) for k, v in param_mean.items()
        }
        self.param_std = {
            k: np.asarray(v, dtype=np.float32) for k, v in param_std.items()
        }
        self.normalize = normalize

        self.sequence_paths: List[Tuple[str, int, str]] = []
        self.param_values = {}
        self._prepare()

    def _prepare(self) -> None:
        start, end = self.frame_range
        for sim_folder in self.sim_folders:
            sim_path = self.root / sim_folder
            desc_path = sim_path / "src" / "description.json"
            params = []
            if desc_path.is_file():
                with open(desc_path, "r") as f:
                    desc = json.load(f)
                for p in self.sim_params:
                    if p == "rey":
                        params.append(float(desc.get("Reynolds Number", 0.0)))
                    else:
                        params.append(0.0)
            else:
                params = [0.0 for _ in self.sim_params]
            self.param_values[sim_folder] = np.array(params, dtype=np.float32)

            max_start = end - (self.conditioning_length + self.target_length)
            for frame in range(start, max_start + 1, self.stride):
                self.sequence_paths.append((str(sim_path), frame, sim_folder))

    def __len__(self) -> int:
        return len(self.sequence_paths)

    def _load_raw_sequence(
        self, sim_path: str, start_frame: int, length: int
    ) -> Optional[np.ndarray]:
        frames = []
        for i in range(length):
            frame_idx = start_frame + i
            channel_list = []
            for field in self.sim_fields:
                fname = f"{FIELD_FILE_MAP[field]}_{frame_idx:06d}.npz"
                fpath = Path(sim_path) / fname
                if not fpath.is_file():
                    return None
                arr = _load_npz(fpath)
                if arr.ndim == 3:
                    # handle both C,H,W and H,W,C conventions
                    if arr.shape[0] <= 3:
                        # already (C,H,W)
                        pass
                    else:
                        # assume (H,W,C)
                        arr = np.transpose(arr, (2, 0, 1))
                else:
                    arr = arr[None, ...]
                channel_list.append(arr.astype(np.float32))
            frame = np.concatenate(channel_list, axis=0)
            frames.append(frame)
        return np.stack(frames, axis=0)

    def normalize_fields(self, fields: np.ndarray) -> np.ndarray:
        mean_fields = np.concatenate([self.field_mean[f] for f in self.sim_fields])
        std_fields = np.concatenate([self.field_std[f] for f in self.sim_fields])

        if fields.ndim == 4:
            return (fields - mean_fields[None, :, None, None]) / std_fields[
                None, :, None, None
            ]
        elif fields.ndim == 3:
            return (fields - mean_fields[:, None, None]) / std_fields[:, None, None]
        else:
            raise ValueError("Unexpected field array shape")

    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        if params.size == 0:
            return params

        mean_params = np.concatenate([self.param_mean[p] for p in self.sim_params])
        std_params = np.concatenate([self.param_std[p] for p in self.sim_params])
        return (params - mean_params) / std_params

    def load_sequence(
        self, sim_folder: str, start_frame: int, target_length: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load a normalized sequence for evaluation or inference."""

        sim_path = os.path.join(self.root, sim_folder)
        length = self.conditioning_length + target_length
        seq = self._load_raw_sequence(sim_path, start_frame, length)

        if seq is None or seq.shape[0] < length:
            return None

        cond_fields = seq[: self.conditioning_length]
        gt_fields = seq[self.conditioning_length : length]
        params = self.param_values.get(sim_folder, np.array([], dtype=np.float32))

        if self.normalize:
            cond_fields = self.normalize_fields(cond_fields)
            gt_fields = self.normalize_fields(gt_fields)
            params = self.normalize_params(params)

        cond_fields = torch.from_numpy(cond_fields)
        gt_fields = torch.from_numpy(gt_fields)
        params_tensor = torch.from_numpy(params)
        T, _, H, W = cond_fields.shape

        if params_tensor.numel() > 0:
            cond_params = (
                params_tensor.view(1, -1, 1, 1)
                .expand(T, -1, H, W)
            )
            conditioning = torch.cat([cond_fields, cond_params], dim=1)

            tgt_params = (
                params_tensor.view(1, -1, 1, 1)
                .expand(gt_fields.shape[0], -1, H, W)
            )
            target = torch.cat([gt_fields, tgt_params], dim=1)
        else:
            conditioning = cond_fields
            target = gt_fields

        return conditioning, target

    def __getitem__(self, idx: int):
        _, start, folder = self.sequence_paths[idx]

        loaded = self.load_sequence(
            sim_folder=folder,
            start_frame=start,
            target_length=self.target_length,
        )

        if loaded is None:
            raise FileNotFoundError("Sequence frames missing")

        return loaded


class PhysicsDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule for physics simulations."""

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        conditioning_length: int = 2,
        dimension: int = 2,
        frame_size: Tuple[int, int] = (128, 64),
        stride: int = 1,
        train_sim_selection: Optional[List[int]] = None,
        test_sim_selection: Optional[List[int]] = None,
        train_frame_range: Tuple[int, int] = (0, 1300),
        test_frame_range: Tuple[int, int] = (1000, 1120),
        sim_fields: Optional[List[str]] = None,
        sim_params: Optional[List[str]] = None,
        field_mean: Optional[Dict[str, List[float]]] = None,
        field_std: Optional[Dict[str, List[float]]] = None,
        param_mean: Optional[Dict[str, List[float]]] = None,
        param_std: Optional[Dict[str, List[float]]] = None,
        target_length: int = 1,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.conditioning_length = conditioning_length
        self.dimension = dimension
        self.frame_size = frame_size
        self.stride = stride
        self.target_length = target_length
        self.train_sim_selection = train_sim_selection or []
        self.test_sim_selection = test_sim_selection or []
        self.train_frame_range = train_frame_range
        self.test_frame_range = test_frame_range
        self.sim_fields = sim_fields or ["vel", "pres"]
        self.sim_params = sim_params or ["rey"]

        if (
            field_mean is None
            or field_std is None
            or param_mean is None
            or param_std is None
        ):
            raise ValueError(
                "Normalization statistics must be provided in the configuration"
            )

        self.field_mean = {k: np.asarray(v, dtype=np.float32) for k, v in field_mean.items()}
        self.field_std = {k: np.asarray(v, dtype=np.float32) for k, v in field_std.items()}
        self.param_mean = {k: np.asarray(v, dtype=np.float32) for k, v in param_mean.items()}
        self.param_std = {k: np.asarray(v, dtype=np.float32) for k, v in param_std.items()}

        self.train_dataset: Optional[PhysicsDataset] = None
        self.test_dataset: Optional[PhysicsDataset] = None

        self.test_sim_folders: List[str] = []


    def setup(self, stage: Optional[str] = None) -> None:
        train_folders = [f"sim_{i:06d}" for i in self.train_sim_selection]
        test_folders = [f"sim_{i:06d}" for i in self.test_sim_selection]

        self.train_dataset = PhysicsDataset(
            self.data_path,
            train_folders,
            self.train_frame_range,
            self.conditioning_length,
            self.stride,
            self.sim_fields,
            self.sim_params,
            self.field_mean,
            self.field_std,
            self.param_mean,
            self.param_std,
            normalize=True,
            target_length=self.target_length,
        )

        self.test_dataset = PhysicsDataset(
            self.data_path,
            test_folders,
            self.test_frame_range,
            self.conditioning_length,
            self.stride,
            self.sim_fields,
            self.sim_params,
            self.field_mean,
            self.field_std,
            self.param_mean,
            self.param_std,
            normalize=True,
            target_length=self.target_length,
        )

        self.test_sim_folders = test_folders


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

