import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yt

# 1. Configuration
base_dir = "../../Exec/Production/2DSwirlCase"

series1_config = {"base": "plt_harps_N2_600W_", "start": 50000, "step": -500}
series2_config = {"base": "plt_harps_N2_1000W_", "start": 38000, "step": -500}

base_name = "600W"
other_name = "1000W"

# Set to 0 or 1 for single-file comparison)
N_FILES = 20

EPSILON = 1e-12
slice_axis = "theta"

plots_config = {
    "temp": {"log_scale": False},
    "y_velocity": {"log_scale": False},
    "x_velocity": {"log_scale": False},
    "AngMom": {"log_scale": False},
    "extsource_rhoh": {"log_scale": False},
    #"Y(O)": {"log_scale": False},
    "Y(N)": {"log_scale": False},
    #"Y(NO)": {"log_scale": False},
    "Y(N2)": {"log_scale": False},
}

# 2. Global Styling Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 13
})

def generate_file_paths(base_path, series_config, count):
    start_idx = series_config["start"]
    step = series_config["step"]
    prefix = series_config["base"]

    num_files = max(1, count)
    indices = [start_idx + i * step for i in range(num_files)]

    file_paths = []
    for idx in indices:
        path = os.path.join(base_path, f"{prefix}{idx}")
        if not os.path.exists(path):
            path = os.path.join(base_path, f"{prefix}{idx:05d}")
        file_paths.append(path)

    return file_paths

def compute_series_average(file_paths, quantity, slice_axis):
    data_sum = None
    bounds = None

    for path in file_paths:
        print(f"  Loading: {path}")
        ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
        slc = yt.SlicePlot(ds, slice_axis, quantity, origin="native")
        slc.swap_axes()
        
        frb_data = np.array(slc.frb[quantity])
        if bounds is None:
            bounds = [float(b) for b in slc.frb.bounds]
            
        if data_sum is None:
            data_sum = np.zeros_like(frb_data, dtype=np.float64)
        data_sum += frb_data

    return data_sum / len(file_paths), bounds

# Generate file lists for both series
files_series1 = generate_file_paths(base_dir, series1_config, N_FILES)
files_series2 = generate_file_paths(base_dir, series2_config, N_FILES)

# Adjust plot config dynamically if HARPS datasets are present
if any("plt_harps" in f for f in files_series1 + files_series2):
    plots_config = {
        "electron_temp": {"log_scale": False},
        "E_over_N": {"log_scale": False},
        "electron_dens": {"log_scale": False}
    }

# 3. Process Quantities and Generate Averaged Difference Plots
for quantity, settings in plots_config.items():
    print(f"\nProcessing quantity: {quantity}")

    print(f"Averaging Series 1 ({len(files_series1)} file(s))...")
    avg1, bounds = compute_series_average(files_series1, quantity, slice_axis)

    print(f"Averaging Series 2 ({len(files_series2)} file(s))...")
    avg2, _ = compute_series_average(files_series2, quantity, slice_axis)

    full_scale = np.max(avg1) - np.min(avg1)

    # Characteristic scale for denominator regularization (5% of total scale range)
    ref_scale = 0.05 * max(full_scale, 1e-6)

    # Differences between ensemble averages: (Avg 2 - Avg 1)
    abs_diff = avg2 - avg1
    rel_diff = (avg2 - avg1) / (np.abs(avg1) + ref_scale)

    # Determine base colormap for average field
    if "Y(" in quantity or "dens" in quantity:
        base_cmap = "viridis"
    elif "velocity" in quantity:
        base_cmap = "RdBu_r"
    elif "AngMom" in quantity or "E_over_N" in quantity:
        base_cmap = "plasma"
    elif "extsource" in quantity:
        base_cmap = "inferno"
    else:
        base_cmap = "magma"

    # 1x3 Subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(10, 7), sharey=True)

    # ----------------------------------------------------
    # Panel 1: Series 1 Mean Field
    # ----------------------------------------------------
    im_avg = axes[0].imshow(
        avg1,
        origin="lower",
        extent=bounds,
        cmap=base_cmap,
        aspect="auto"
    )
    axes[0].set_title("Base Average\n($\overline{\mathrm{" + base_name + "}}$)")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel("z [m]")

    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.08)
    cbar0 = fig.colorbar(im_avg, cax=cax0)
    cbar0.set_label(f"{quantity}", rotation=270, labelpad=15)

    # ----------------------------------------------------
    # Panel 2: Absolute Difference
    # ----------------------------------------------------
    max_abs = np.max(np.abs(abs_diff))
    max_abs = max_abs if max_abs > 0 else 1.0

    im_abs = axes[1].imshow(
        abs_diff,
        origin="lower",
        extent=bounds,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        aspect="auto"
    )
    axes[1].set_title("Absolute Difference\n($\overline{\mathrm{" + other_name + "}} - \overline{\mathrm{" + base_name + "}}$)")
    axes[1].set_xlabel("r [m]")

    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.08)
    cbar1 = fig.colorbar(im_abs, cax=cax1)
    cbar1.set_label(f"$\Delta$ {quantity}", rotation=270, labelpad=15)

    # ----------------------------------------------------
    # Panel 3: Relative Difference
    # ----------------------------------------------------
    max_rel = np.max(np.abs(rel_diff))
    vlim_rel = min(max_rel, 1.0) if max_rel > 0 else 1.0

    im_rel = axes[2].imshow(
        rel_diff,
        origin="lower",
        extent=bounds,
        cmap="twilight",
        vmin=-vlim_rel,
        vmax=vlim_rel,
        aspect="auto"
    )
    axes[2].set_title("Relative Difference\n(Fractional)")
    axes[2].set_xlabel("r [m]")

    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.08)
    cbar2 = fig.colorbar(im_rel, cax=cax2)
    cbar2.set_label("Fractional ($\Delta / |\overline{A}|$)", rotation=270, labelpad=15)

    num_str = f"{len(files_series1)} files avg" if len(files_series1) > 1 else "single file"
    fig.suptitle(f"Averaged Comparison ({num_str}): {quantity}", y=0.98, fontweight="bold")
    plt.tight_layout()

    output_filename = f"diff_avg_{quantity}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot as: {output_filename}")