import numpy as np
from ase.io import read, write
from scipy.spatial.transform import Rotation


def align_trajectory(arg,  atom_indices=None, ref_atom_index=120, rmsd_interval=1):
    # Read the trajectory file
    trajectory = read(arg, index=':')
    # Select the reference frame
    reference_frame = trajectory[0]
    # Use the specified reference atom for alignment
    ref_com = reference_frame.get_positions()[ref_atom_index]
    ref_positions = reference_frame.get_positions()
    cell_center = reference_frame.get_cell().sum(axis=0) / 2
    # Iterate through all frames and align them to the reference frame
    for frame in trajectory:
        frame_com = frame.get_positions()[ref_atom_index]
        frame_positions = frame.get_positions()

        # Calculate the rotation matrix
        r = Rotation.align_vectors(frame_positions - frame_com, ref_positions - ref_com)[0]
        rotation_matrix = r.as_matrix()

        # Apply the rotation
        aligned_positions = np.dot(frame_positions - frame_com, rotation_matrix) + ref_com
        frame.set_positions(aligned_positions)
        # Wrap positions into the central cell box
        # Center the structure using the middle of the cell
        shift = cell_center - np.mean(aligned_positions, axis=0)
        frame.translate(shift)
        # Calculate RMSD
        # Resample the RMSD over intervals
    rmsd_values_interval = []
    for i in range(0, len(trajectory), rmsd_interval):
        # Select frames in the current interval
        interval_frames = trajectory[i:i + rmsd_interval]
        reference_frame = trajectory[0]
        ref_positions = reference_frame.get_positions()
        # Calculate RMSD for selected atoms in the interval and average over them
        interval_rmsds = []
        for frame in interval_frames:
            frame_positions = frame.get_positions()

            # If atom_indices is provided, calculate RMSD only for those atoms
            if atom_indices is not None:
                rmsd = np.sqrt(np.mean((frame_positions[atom_indices] - ref_positions[atom_indices]) ** 2))
            else:
                rmsd = np.sqrt(np.mean((frame_positions - ref_positions) ** 2))
            interval_rmsds.append(rmsd)

        # Average RMSD for this interval
        avg_rmsd = np.mean(interval_rmsds)
        rmsd_values_interval.append(avg_rmsd)
    print(len(rmsd_values_interval))
    # Write the aligned trajectory
    write(arg + '/alinged_trajectory.xyz', trajectory)


    time_points = np.arange(0, len(rmsd_values_interval) * rmsd_interval, rmsd_interval)
    return trajectory, time_points, rmsd_values_interval,rmsd_interval


def calculate_rmsd_for_selection(trajectory, ref_positions, atom_indices):

    rmsd_values = []
    for frame in trajectory:
        frame_positions = frame.get_positions()
        rmsd = np.sqrt(np.mean((frame_positions[atom_indices] - ref_positions[atom_indices]) ** 2))
        rmsd_values.append(rmsd)
    return rmsd_values

from pylab import *
from scipy.stats import linregress

def default_colors(num_colors):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_markers = ['-', '-', '-', '-']
    return color_cycle[:num_colors], custom_markers[:num_colors]

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlabel("Time (ps)", fontsize=16)
ax.set_ylabel('RMSDs ($\AA$)', fontsize=16)
spine_width = 1.5
ax.spines['top'].set_linewidth(spine_width)
ax.spines['right'].set_linewidth(spine_width)
ax.spines['bottom'].set_linewidth(spine_width)
ax.spines['left'].set_linewidth(spine_width)
if __name__ == "__main__":
    if len(sys.argv) >= 2:  # Single argument
        arg = sys.argv[1]
        colors, markers = default_colors(1)
        color = colors[0]
        print("Enter the atom indices or range to calculate RMSD (e.g., 0,1,2 or 0-10):")
        selection = input("Selection: ").strip()

        if "-" in selection:
            start, end = map(int, selection.split("-"))
            atom_indices = np.arange(start, end + 1)
        else:
            atom_indices = list(map(int, selection.split(",")))

        # Call the alignment function with atom selection
        trajectory, tp, rmsds,intv = align_trajectory(arg, atom_indices=atom_indices)

        plt.plot(tp/1000,rmsds, linestyle="-", color=color)
        #ax.legend(prop={'size': 16})

plt.savefig('MD_RMSDs_time.png')
plt.show()
