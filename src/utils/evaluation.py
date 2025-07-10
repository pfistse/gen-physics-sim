import torch
import numpy as np
from typing import Dict, Any, List
from tqdm.auto import tqdm
import pytorch_lightning as pl
from utils.log import get_logger
from utils import metrics
from utils.wandb import log_samples_video

logger = get_logger("utils.eval")


def calculate_temporal_deviation(
    model: pl.LightningModule,
    num_simulations: int = 10,
    num_time_steps: int = 20,
    start_frame: int = None,
    channel_titles: List[str] = None,
    **generation_kwargs
) -> Dict[str, Any]:
    """Calculate deviation from ground truth physics data for generated sequences."""
    model.eval()
    device = next(model.parameters()).device
    
    datamodule = model.trainer.datamodule
    test_sim_folders = datamodule.test_sim_folders[:num_simulations]

    if start_frame is None:
        start_frame = datamodule.test_dataset.frame_range[0]
    
    # Prepare conditioning and ground truth data
    sequences, ground_truths = [], []
    for sim_folder in test_sim_folders:
        data = datamodule.test_dataset.load_sequence(
            sim_folder=sim_folder,
            start_frame=start_frame,
            target_length=num_time_steps,
        )

        if data is None:
            continue

        conditioning, gt_fields = data

        sequences.append(conditioning)
        ground_truths.append(gt_fields)

    # Generate sequences and calculate metrics
    all_mse, all_rmse = [], []
    results = []
    
    with torch.no_grad():
        for i, (conditioning, gt_sequence) in enumerate(
            tqdm(zip(sequences, ground_truths), total=len(sequences), desc="eval", leave=False)
        ):
            # Generate sequence
            generated = model.generate_sequence_samples(
                conditioning.unsqueeze(0).to(device),
                num_frames=num_time_steps,
                **generation_kwargs
            ).squeeze(0)  # [T, C_total, H, W]

            gt_sequence = gt_sequence.to(device)  # [T, C_fields, H, W]

            # Calculate frame-by-frame metrics
            field_channels = gt_sequence.shape[1] - len(datamodule.sim_params)
            generated_fields = generated[:, :field_channels].contiguous()  # Extract field channels only
            gt_sequence = gt_sequence[:, :field_channels].contiguous()
            
            mse_per_step = metrics.mean_squared_error(generated_fields, gt_sequence).cpu().tolist()
            rmse_per_step = metrics.root_mean_squared_error(generated_fields, gt_sequence).cpu().tolist()

            all_mse.append(mse_per_step)
            all_rmse.append(rmse_per_step)

            results.append({
                'simulation_index': i,
                'mse_per_step': mse_per_step,
                'rmse_per_step': rmse_per_step,
                'mse_mean': np.mean(mse_per_step),
                'rmse_mean': np.mean(rmse_per_step),
            })

    # Aggregate results
    all_mse = np.array(all_mse)
    all_rmse = np.array(all_rmse)

    return {
        'mse_per_timestep': np.mean(all_mse, axis=0).tolist(),
        'rmse_per_timestep': np.mean(all_rmse, axis=0).tolist(),
        'mse_mean': float(np.mean(all_mse)),
        'rmse_mean': float(np.mean(all_rmse)),
        'simulations': results,
    }
