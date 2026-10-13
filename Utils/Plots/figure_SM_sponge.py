import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yt

base_dir = "../../Exec/Production/2DSwirlCase"
series1_config = {"base": "plt_no_sponge_", "start": 43000, "step": -1000}
series2_config = {"base": "plt_sponge_", "start": 43000, "step": -1000}
N_FILES = 10
slice_axis = "theta"
selected_quantities = ["AngMom", "temp", "y_velocity", "Y(N)"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

def generate_file_paths(base_path, series_config, count):
    indices = [series_config["start"] + i * series_config["step"] for i in range(max(1, count))]
    file_paths = []
    for idx in indices:
        path = os.path.join(base_path, f"{series_config['base']}{idx}")
        if not os.path.exists(path):
            path = os.path.join(base_path, f"{series_config['base']}{idx:05d}")
        file_paths.append(path)
    return file_paths

def compute_series_average(file_paths, quantity, slice_axis):
    data_sum = None
    bounds = None
    for path in file_paths:
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

files_series1 = generate_file_paths(base_dir, series1_config, N_FILES)
files_series2 = generate_file_paths(base_dir, series2_config, N_FILES)

abs_diffs = {}
rel_diffs = {}
bounds_cm = None

for q in selected_quantities:
    avg1, b = compute_series_average(files_series1, q, slice_axis)
    if bounds_cm is None:
        bounds_cm = [val * 100.0 for val in b]  # Convert meters to centimeters
    avg2, _ = compute_series_average(files_series2, q, slice_axis)
    
    # Scale Y(N) to percentage
    if q == "Y(N)":
        avg1_scaled = avg1 * 100.0
        avg2_scaled = avg2 * 100.0
    else:
        avg1_scaled = avg1
        avg2_scaled = avg2

    full_scale = np.max(avg1_scaled) - np.min(avg1_scaled)
    ref_scale = 0.05 * max(full_scale, 1e-6)
    
    abs_diffs[q] = avg2_scaled - avg1_scaled
    rel_diffs[q] = (avg2_scaled - avg1_scaled) / (np.abs(avg1_scaled) + ref_scale)

prop_map = {
    "AngMom": {"sym": "l_z", "unit": "", "cmap": "RdBu_r"},
    "temp": {"sym": "T_g", "unit": r"\mathrm{K}", "cmap": "RdGy_r"},
    "y_velocity": {"sym": "u_z", "unit": r"\mathrm{m/s}", "cmap": "seismic"},
    "Y(N)": {"sym": "Y_{\mathrm{N}}", "unit": r"\%", "cmap": "PiYG"},
}

def plot_comparison(diff_dict, is_relative, filename_base):
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharex=True, sharey=True)

    for idx, q in enumerate(selected_quantities):
        ax = axes[idx]
        diff_data = diff_dict[q]
        max_val = np.max(np.abs(diff_data))
        max_val = max_val if max_val > 0 else 1.0

        cfg = prop_map[q]
        im = ax.imshow(
            diff_data,
            origin="lower",
            extent=bounds_cm,
            cmap=cfg["cmap"],
            vmin=-max_val,
            vmax=max_val,
            aspect="auto"
        )

        ax.set_xlabel(r"$r\ (\mathrm{cm})$")
        if idx == 0:
            ax.set_ylabel(r"$z\ (\mathrm{cm})$")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)

        sym = cfg["sym"]
        unit = cfg["unit"]
        if is_relative:
            cbar.set_label(f"Relative $\\Delta {sym}$", rotation=270, labelpad=12)
        else:
            if unit:
                cbar.set_label(f"$\\Delta {sym}\\ [{unit}]$", rotation=270, labelpad=12)
            else:
                cbar.set_label(f"$\\Delta {sym}$", rotation=270, labelpad=12)

    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{filename_base}.pdf", bbox_inches="tight")
    plt.close(fig)

plot_comparison(abs_diffs, False, "sponge_comparison_absolute")
plot_comparison(rel_diffs, True, "sponge_comparison_relative")