import numpy as np
from ase.io import read
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ase import Atoms
from ase.geometry import get_distances
from scipy.stats import linregress

def align_trajectory(inputs, interval=500):
    # Unpack the inputs (arg, ind1, ind2)
    arg, ind1, ind2 = inputs

    # Read the trajectory file
    trajectory = read(arg + '/XDATCAR', index=':')

    # Select the reference frame
    reference_frame = trajectory[0]
    cell = trajectory[0].get_cell()
    # Extract positions of selected atoms in the reference frame
    ref_positions = reference_frame.get_positions(wrap=True)
    # Initialize MSD values list
    msd_values = []
    all_positions = []
    for i in range(0, len(trajectory)):
        frame = trajectory[i]
        frame_positions = frame.get_positions(wrap=True)
        all_positions.append(frame_positions)
    #com_positions = np.mean(all_positions, axis=1)  # COM for each frame
    #drift_removed_positions = all_positions - com_positions[:, np.newaxis, :]
    all_positions = np.array(all_positions)
    selected_positions = all_positions[:,ind1:ind2,:]


    #selected_positions = drift_removed_positions[:, ind1:ind2]

    for lag in range(1,len(trajectory),interval):
        displacements = selected_positions[lag:] - selected_positions[:-lag]
        displacements -= np.mean(displacements, axis=1, keepdims=True)
        msd = np.mean(np.sum(displacements ** 2, axis=-1))
        msd_values.append(msd)


    tau_list = np.arange(1,len(trajectory),interval)
    slope, intercept, r_value, p_value, std_err = linregress(tau_list, msd_values)
    diffusion_coefficient = slope / (2 * 3)

    return tau_list, msd_values, diffusion_coefficient, std_err

def default_colors(num_colors):
    # Define a default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    while len(color_cycle) < num_colors:
        color_cycle += color_cycle  # Repeat colors if needed
    return color_cycle[:num_colors]


def main():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Time (ps)", fontsize=16)
    ax.set_ylabel('MSD ($\AA^2$)', fontsize=16)
    spine_width = 1.5
    ax.spines['top'].set_linewidth(spine_width)
    ax.spines['right'].set_linewidth(spine_width)
    ax.spines['bottom'].set_linewidth(spine_width)
    ax.spines['left'].set_linewidth(spine_width)

    # Parse command-line arguments
    arguments = sys.argv[1:]
    if len(arguments) % 3 != 0:
        raise ValueError("Arguments must be a multiple of 3: <path> <ind1> <ind2> for each group.")

    # Group arguments into sets of 3 (arg, ind1, ind2)
    grouped_arguments = [
        (arguments[i], int(arguments[i + 1]), int(arguments[i + 2]))
        for i in range(0, len(arguments), 3)
    ]
    colors = default_colors(len(grouped_arguments))
    for i, inputs in enumerate(grouped_arguments):
        tp, msds, diff,std = align_trajectory(inputs)
        color = colors[i]
        label = f"Group {i + 1}: Atoms {inputs[1]}-{inputs[2] - 1}"
        plt.plot(tp / 1000 * 2, msds, linestyle='-', color=color, label=label)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

