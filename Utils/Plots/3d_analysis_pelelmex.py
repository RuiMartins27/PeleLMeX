import os
from pathlib import Path
import re
import numpy as np
import yt


base_dir = "../../Exec/Production/3D_Microwave"
slice_axes = ["y"]  #  ["x", "y", "z"]
z_slice_center = 0.281  # Center for z-slice (0.361 = inlet, 0.280 = WG center)
use_all_plt = True

cfd_plots_config = {
    #"extsource_rhoh": {"log_scale": False, "fix_scale": False},
    "z_velocity": {"log_scale": False, "fix_scale": True},
    #"mag_vel": {"log_scale": False, "fix_scale": False},
    # "x_velocity": {"log_scale": False, "fix_scale": True},
    "temp": {"log_scale": False, "fix_scale": False},
    "X(O)": {"log_scale": False, "fix_scale": False},
    "X(N2)": {"log_scale": False, "fix_scale": False},
    #"X(N)": {"log_scale": False, "fix_scale": False},
    # "density": {"log_scale": False, "fix_scale": False},
}

z_limits = {
    "temp": (275, 6050),
    "z_velocity": (-5, 5),
    "y_velocity": (-5.0, 5.0),
    "x_velocity": (-5.0, 5.0),
    "extsource_rhoh": (1e-4, 0.5),
}


def get_all_plt_dirs(base_dir):
    base_path = Path(base_dir)
    plt_dirs = [p for p in base_path.glob("plt_*") if p.is_dir()]

    def extract_step(path):
        match = re.search(r"plt_(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return sorted(
        [str(p) for p in plt_dirs if extract_step(p) != -1],
        key=lambda p: extract_step(Path(p)),
    )


def get_colormap(quantity):
    if any(k in quantity for k in ("Y(", "dens", "X(")):
        return "viridis"
    elif "velocity" in quantity:
        return "RdBu_r"
    elif "extsource" in quantity:
        return "inferno"
    elif "temp" in quantity:
        return "magma"
    elif "E_over_N" in quantity:
        return "plasma"
    return "CMRmap"


def generate_averaged_slice_plots(plt_folders, config, axes, prefix="3d_slice_avg"):
    if not plt_folders:
        print(f"[{prefix}] Error: No plotfiles found.")
        return

    # Check field existence using the first dataset as reference
    sample_ds = yt.load(plt_folders[0], units_override={"length_unit": (1.0, "m")})
    available_fields = [f[1] for f in sample_ds.field_list] + [f[1] for f in sample_ds.derived_field_list]

    for slice_axis in axes:
        for quantity, settings in config.items():
            if quantity not in available_fields:
                print(f"[{prefix}] Skipping {quantity}, not found in dataset.")
                continue

            print(
                f"[{prefix}] Averaging {quantity} along {slice_axis}-axis across {len(plt_folders)} frames..."
            )

            frb_arrays = []
            last_slc = None

            for folder in plt_folders:
                ds = yt.load(folder, units_override={"length_unit": (1.0, "m")})

                if slice_axis == "z":
                    slc = yt.SlicePlot(ds, slice_axis, quantity, center=[0.0, 0.0, z_slice_center])
                else:
                    slc = yt.SlicePlot(ds, slice_axis, quantity, origin="native")

                frb_arrays.append(np.array(slc.frb[quantity]))
                last_slc = slc

            stacked = np.stack(frb_arrays)
            avg_values = stacked.mean(axis=0)

            # Unpack settings here so they are available
            log_scale = settings["log_scale"]
            fix_scale = settings["fix_scale"]

            # Determine color limits
            if fix_scale and quantity in z_limits:
                z_min, z_max = z_limits[quantity]
                if log_scale and z_min <= 0:
                    z_min = 1e-6
            else:
                z_min, z_max = float(np.nanmin(avg_values)), float(np.nanmax(avg_values))
                if log_scale and z_min <= 0:
                    z_min = 1e-6

            # 1. Apply all yt styling FIRST
            if log_scale and not fix_scale:
                last_slc.set_log(quantity, True, linthresh=1e-6)
            else:
                last_slc.set_log(quantity, log_scale)

            last_slc.set_zlim(quantity, z_min, z_max)

            cmap = get_colormap(quantity)
            last_slc.set_cmap(quantity, cmap)
            last_slc.set_font({"size": 12, "family": "serif"})

            # 2. Force yt to construct the Matplotlib plots behind the scenes
            last_slc._setup_plots()

            # 3. OVERWRITE the matplotlib image array with the averaged data
            plot = last_slc.plots[quantity]
            plot.image.set_data(avg_values)

            # 4. Save the figure (won't overwrite data since plot is marked valid)
            out_name = f"analysis_outputs/{prefix}_{slice_axis}_{quantity}.png"
            last_slc.save(out_name)
            print(f"[{prefix}] Saved: {out_name}")


if __name__ == "__main__":
    if use_all_plt:
        plt_folders = get_all_plt_dirs(base_dir)
    else:
        base_step = 280000
        step_size = 500
        n_folders = 10

        steps = [base_step - i * step_size for i in range(n_folders)] 
        plt_folders = [f"{base_dir}/plt_{step}" for step in steps]

    if plt_folders:
        print(f"Found {len(plt_folders)} plotfiles in {base_dir}: {[Path(p).name for p in plt_folders]}")
        generate_averaged_slice_plots( plt_folders, cfd_plots_config, axes=slice_axes)
    else:
        print(f"Error: No plt_* folders found inside {base_dir}")