#!/usr/bin/env python3
"""Generate random simulation data matching the expected layout."""

import os
import numpy as np
import json
import shutil
from pathlib import Path

def create_dummy_simulation_data(
    output_dir="data/dummy_128_inc", 
    num_simulations=20, 
    num_timesteps=1300
):
    """
    Generate dummy simulation data matching the original structure.
    
    Args:
        output_dir: Directory to create the dummy data in
        num_simulations: Number of simulations to generate (default 20)
        num_timesteps: Number of timesteps per simulation (default 1300)
    """
    
    # Configuration based on the real data structure
    frame_height = 64  # From config frame_size
    frame_width = 128  # From config frame_size
    warmup_steps = 20
    dt = 0.05
    
    # Simulation parameters (these will vary slightly between simulations)
    base_reynolds = 1000.0
    base_velocity = 0.5
    base_viscosity = 0.0003
    cylinder_size = 0.6
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating dummy data in: {output_dir}")
    
    for sim_idx in range(num_simulations):
        sim_folder = f"sim_{sim_idx:06d}"
        sim_path = os.path.join(output_dir, sim_folder)
        os.makedirs(sim_path, exist_ok=True)
        
        print(f"Generating simulation {sim_idx + 1}/{num_simulations}: {sim_folder}")
        
        # Create subfolders
        render_path = os.path.join(sim_path, "render")
        src_path = os.path.join(sim_path, "src")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(src_path, exist_ok=True)
        
        # Generate slightly different parameters for each simulation
        reynolds = base_reynolds + np.random.uniform(-100, 100)
        velocity = base_velocity + np.random.uniform(-0.1, 0.1)
        viscosity = velocity * cylinder_size / reynolds
        
        # Generate obstacle mask (static for all timesteps)
        obstacle_mask = generate_obstacle_mask(frame_width, frame_height, cylinder_size)
        np.savez_compressed(
            os.path.join(sim_path, "obstacle_mask.npz"),
            data=obstacle_mask
        )
        
        # Generate velocity and pressure data for all timesteps
        for timestep in range(num_timesteps):
            if timestep % 100 == 0:
                print(f"  Generating timestep {timestep}/{num_timesteps}")
                
            # Generate realistic-looking velocity field (2D vector field)
            velocity_field = generate_velocity_field(frame_width, frame_height, obstacle_mask, velocity, timestep)
            
            # Generate pressure field (scalar field)
            pressure_field = generate_pressure_field(frame_width, frame_height, obstacle_mask, timestep)
            
            # Save velocity and pressure files
            np.savez_compressed(
                os.path.join(sim_path, f"velocity_{timestep:06d}.npz"),
                data=velocity_field
            )
            
            np.savez_compressed(
                os.path.join(sim_path, f"pressure_{timestep:06d}.npz"),
                data=pressure_field
            )
        
        # Create context.json
        context_data = {
            "phi_version": "2.0.3",
            "argv": ["demos/karman.py", str(frame_width*2), str(frame_height*2), str(dt), 
                    str(num_timesteps), str(warmup_steps), str(cylinder_size), 
                    str(velocity), "-1", str(int(reynolds))]
        }
        
        with open(os.path.join(src_path, "context.json"), "w") as f:
            json.dump(context_data, f)
        
        # Create description.json with simulation statistics
        description_data = create_description_json(
            frame_width*2, frame_height*2, dt, num_timesteps, warmup_steps,
            cylinder_size, velocity, viscosity, reynolds
        )
        
        with open(os.path.join(src_path, "description.json"), "w") as f:
            json.dump(description_data, f, indent=4)
        
        # Create a dummy karman.py file
        karman_content = f'''#!/usr/bin/env python3
"""
Dummy Karman vortex street simulation script.
This is a placeholder file - generated dummy data.
"""

# Simulation parameters
RESOLUTION = [{frame_width*2}, {frame_height*2}]
DT = {dt}
STEPS = {num_timesteps}
WARMUP = {warmup_steps}
CYLINDER_SIZE = {cylinder_size}
VELOCITY = {velocity}
VISCOSITY = {viscosity}
REYNOLDS = {reynolds}

print("This is dummy simulation data - not a real PhiFlow simulation")
'''
        
        with open(os.path.join(src_path, "karman.py"), "w") as f:
            f.write(karman_content)
        
        # Create a dummy video file name in render folder
        video_name = f"cyl{cylinder_size:.2f}_vel{velocity:.2f}_visc{viscosity:.8f}_rey{int(reynolds):06d}.mp4"
        dummy_video_path = os.path.join(render_path, video_name)
        
        # Create a tiny dummy file to represent the video
        with open(dummy_video_path, "w") as f:
            f.write("dummy video file")
    
    print(f"\nGenerated {num_simulations} dummy simulations in {output_dir}")
    print(f"Total files created: {num_simulations * (num_timesteps * 2 + 1 + 3 + 1)} files")


def generate_obstacle_mask(width, height, cylinder_size):
    """Generate a circular obstacle mask in the flow field."""
    # Create coordinate grids
    x = np.linspace(0, 4, width)  # Domain from 0 to 4
    y = np.linspace(0, 2, height)  # Domain from 0 to 2
    X, Y = np.meshgrid(x, y)
    
    # Cylinder center (typical position for Karman vortex street)
    center_x = 1.0
    center_y = 1.0
    radius = cylinder_size * 0.5
    
    # Create circular obstacle
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    obstacle = (distance <= radius).astype(np.float32)
    
    return obstacle


def generate_velocity_field(width, height, obstacle_mask, inlet_velocity, timestep):
    """Generate a realistic-looking velocity field with flow around obstacle."""
    # Create base flow
    x = np.linspace(0, 4, width)
    y = np.linspace(0, 2, height)
    X, Y = np.meshgrid(x, y)
    
    # Base horizontal flow
    u = np.full_like(X, inlet_velocity, dtype=np.float32)
    v = np.zeros_like(X, dtype=np.float32)
    
    # Add some time-varying vortex shedding behind the cylinder
    if timestep > 20:  # After warmup
        # Simple sinusoidal variation to simulate vortex shedding
        freq = 0.1  # Frequency of vortex shedding
        phase = 2 * np.pi * freq * timestep
        
        # Add perturbations downstream of cylinder
        downstream_mask = (X > 1.5) & (X < 3.5)
        
        # Oscillating flow
        amplitude = 0.3 * inlet_velocity
        u[downstream_mask] += amplitude * np.sin(phase + Y[downstream_mask] * 2) * np.exp(-(X[downstream_mask] - 1.5) * 0.5)
        v[downstream_mask] += amplitude * np.cos(phase + Y[downstream_mask] * 2) * np.exp(-(X[downstream_mask] - 1.5) * 0.5)
    
    # Add some random noise
    noise_scale = 0.05 * inlet_velocity
    u += np.random.normal(0, noise_scale, u.shape).astype(np.float32)
    v += np.random.normal(0, noise_scale, v.shape).astype(np.float32)
    
    # Zero velocity inside obstacle
    u[obstacle_mask.astype(bool)] = 0
    v[obstacle_mask.astype(bool)] = 0
    
    # Stack u and v components (shape: [height, width, 2])
    velocity = np.stack([u, v], axis=-1)
    
    return velocity


def generate_pressure_field(width, height, obstacle_mask, timestep):
    """Generate a pressure field around the obstacle."""
    x = np.linspace(0, 4, width)
    y = np.linspace(0, 2, height)
    X, Y = np.meshgrid(x, y)
    
    # Create pressure variation around cylinder
    center_x = 1.0
    center_y = 1.0
    
    # Higher pressure upstream, lower downstream
    pressure = np.zeros_like(X, dtype=np.float32)
    
    # Base pressure gradient
    pressure += 0.5 - 0.1 * X  # Decreasing pressure in flow direction
    
    # Pressure perturbation around cylinder
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # High pressure at stagnation point (upstream)
    upstream_mask = (X < center_x) & (distance < 1.0)
    pressure[upstream_mask] += 0.2 * np.exp(-distance[upstream_mask] * 2)
    
    # Low pressure in wake (downstream)
    downstream_mask = (X > center_x) & (distance < 1.5)
    pressure[downstream_mask] -= 0.15 * np.exp(-distance[downstream_mask])
    
    # Add time-varying component for vortex shedding
    if timestep > 20:
        freq = 0.1
        phase = 2 * np.pi * freq * timestep
        wake_variation = 0.1 * np.sin(phase + Y * 3) * np.exp(-(X - center_x) * 0.8)
        pressure[X > center_x] += wake_variation[X > center_x]
    
    # Add noise
    noise_scale = 0.02
    pressure += np.random.normal(0, noise_scale, pressure.shape).astype(np.float32)
    
    return pressure


def create_description_json(width, height, dt, steps, warmup, cylinder_size, velocity, viscosity, reynolds):
    """Create a description.json file with simulation metadata."""
    
    # Generate some dummy statistics
    velocity_stats = []
    pressure_stats = []
    
    for step in range(min(50, steps)):  # Just generate first 50 entries
        if step == 0:
            vel_stat = f"Min:0.00000000 Max:{velocity:.8f} Avg: {velocity/2:.8f}"
        else:
            # Simulate some variation
            min_vel = -0.4 + np.random.uniform(-0.1, 0.1)
            max_vel = velocity * (1.5 + np.random.uniform(-0.2, 0.2))
            avg_vel = velocity * (0.4 + np.random.uniform(-0.1, 0.1))
            vel_stat = f"Min:{min_vel:.8f} Max:{max_vel:.8f} Avg: {avg_vel:.8f}"
        
        velocity_stats.append(vel_stat)
        
        # Similar for pressure
        min_pres = -0.5 + np.random.uniform(-0.1, 0.1)
        max_pres = 0.8 + np.random.uniform(-0.1, 0.1)
        avg_pres = 0.1 + np.random.uniform(-0.05, 0.05)
        pres_stat = f"Min:{min_pres:.8f} Max:{max_pres:.8f} Avg: {avg_pres:.8f}"
        pressure_stats.append(pres_stat)
    
    description = {
        "Timestamp": "2025-06-10 12:00:00",  # Current date
        "Resolution": [width, height],
        "Dt": dt,
        "Steps, Warmup": [steps, warmup],
        "Cylinder Size": cylinder_size,
        "Walls (lrtb)": [0.7, 2.7, 0.7, 0.7],  # Standard wall positions
        "Inflow Velocity": velocity,
        "Fluid Viscosity": viscosity,
        "Reynolds Number": reynolds,
        "Stats": {
            "Velocity": velocity_stats,
            "Pressure": pressure_stats
        }
    }
    
    return description


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dummy physics simulation data")
    parser.add_argument("--output-dir", default="data/dummy_128_inc", 
                       help="Output directory for dummy data")
    parser.add_argument("--num-sims", type=int, default=20, 
                       help="Number of simulations to generate")
    parser.add_argument("--num-timesteps", type=int, default=1300,
                       help="Number of timesteps per simulation")
    
    args = parser.parse_args()
    
    print("Generating dummy physics simulation data...")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of simulations: {args.num_sims}")
    print(f"Timesteps per simulation: {args.num_timesteps}")
    
    create_dummy_simulation_data(
        args.output_dir, 
        args.num_sims, 
        args.num_timesteps
    )
    
    print("\nDummy data generation complete!")
    print("You can now use this data for testing your physics simulation pipeline.")
