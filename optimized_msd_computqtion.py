import numpy as np
import matplotlib.pyplot as plt
import tidynamics
from ase.io import read
import sys

def compute_msd_optimized(traj_path, atom_start, atom_end, time_step_fs=2):
    try:
        traj = read(traj_path, index=":")
    except Exception as e:
        print(f"Error reading trajectory {traj_path}: {e}")
        return None, None

    n_frames = len(traj)
    n_atoms = len(traj[0])

    if atom_end > n_atoms:
        print(f"Error: Atom range [{atom_start}, {atom_end}) exceeds system size ({n_atoms} atoms).")
        return None, None

    positions = np.array([atoms.positions[atom_start:atom_end] for atoms in traj])
    positions = np.swapaxes(positions, 0, 1)  # Shape: (n_selected, n_frames, 3)

    msd_x = np.mean([tidynamics.msd(atom_positions[:, 0]) for atom_positions in positions], axis=0)
    msd_y = np.mean([tidynamics.msd(atom_positions[:, 1]) for atom_positions in positions], axis=0)
    msd_z = np.mean([tidynamics.msd(atom_positions[:, 2]) for atom_positions in positions], axis=0)

    msd_values = msd_x + msd_y + msd_z
    lag_times = np.arange(1, len(msd_values) + 1) * (time_step_fs / 1000.0)  # Convert fs to ps

    return lag_times, msd_values

# Create a single figure before looping
plt.figure(figsize=(8, 6))

if (len(sys.argv) - 1) % 3 != 0:
    print("Error: Incorrect number of arguments. Expecting multiples of 3 (traj_path, atom_start, atom_end).")
    sys.exit(1)

# Loop through inputs and plot all MSDs in the same figure
for i in range(1, len(sys.argv), 3):
    traj_path = sys.argv[i]
    atom_start = int(sys.argv[i+1])
    atom_end = int(sys.argv[i+2])
    
    lag_times, msd_values = compute_msd_optimized(traj_path, atom_start, atom_end)
    if msd_values is not None:
        plt.plot(lag_times, msd_values, label=f"MSD {traj_path}")

# Finalize plot after all curves are added
plt.xlabel("Time (ps)", fontsize=16)
plt.ylabel("MSD ($\AA^2$)", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
