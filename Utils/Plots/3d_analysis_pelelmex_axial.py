import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yt

base_dir = "../../Exec/Production/3D_Microwave"
x_probe = 0.0
y_probe = 0.0
z_min = 0.0
z_max = 0.3
n_z_points = 200

cfd_plots_config = {
    "extsource_rhoh": {"log_scale": False, "fix_scale": False},
    "x_velocity": {"log_scale": False, "fix_scale": True},
    "y_velocity": {"log_scale": False, "fix_scale": False},
    "z_velocity": {"log_scale": False, "fix_scale": False},
    "mag_vel": {"log_scale": False, "fix_scale": False},
    "temp": {"log_scale": False, "fix_scale": False},
}

harps_plots_config = {
    "electron_temp": {"log_scale": False, "fix_scale": False},
    "E_field": {"log_scale": False, "fix_scale": False},
    "E_over_N": {"log_scale": False, "fix_scale": False},
    "electron_dens": {"log_scale": False, "fix_scale": False},
}

def get_plt_folders(base_dir, prefix, base_step, step_size, n_folders):
    folders = []
    for i in range(n_folders):
        step = base_step - i * step_size
        path = os.path.join(base_dir, f"{prefix}{step:05d}")
        if os.path.exists(path): folders.append(path)
    return folders

def extract_z_profile(ds, field):
    ray = ds.ray([x_probe, y_probe, z_min], [x_probe, y_probe, z_max])
    sort_idx = np.argsort(ray["index", "z"].v)
    z_coords = ray["index", "z"][sort_idx].v
    vals = ray[field][sort_idx].vn
    z_grid = np.linspace(z_min, z_max, n_z_points)
    return z_grid, np.interp(z_grid, z_coords, vals)

def generate_z_profile_plots(plt_folders, config, prefix="3d_z_profile"):
    if not plt_folders: print(f"[{prefix}] Error: No plotfiles found."); return
    sample_ds = yt.load(plt_folders[0], units_override={"length_unit": (1.0, "m")})
    available_fields = [f[1] for f in sample_ds.field_list] + [f[1] for f in sample_ds.derived_field_list]
    os.makedirs("analysis_outputs", exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'mathtext.fontset': 'stix'})
    for quantity, settings in config.items():
        if quantity not in available_fields: continue
        print(f"[{prefix}] Processing {quantity} across {len(plt_folders)} frames...")
        profile_arrays = []
        z_grid = None
        for folder in plt_folders:
            ds = yt.load(folder, units_override={"length_unit": (1.0, "m")})
            zg, vals = extract_z_profile(ds, quantity)
            z_grid = zg
            profile_arrays.append(vals)
        avg_profile = np.stack(profile_arrays).mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(avg_profile, z_grid, color="tab:blue", linewidth=1.5)
        ax.set_ylabel("Z Coordinate (m)", fontsize=14, labelpad=6)
        ax.set_xlabel(quantity, fontsize=14, labelpad=6)
        if settings["log_scale"]: ax.set_xscale("log")
        ax.minorticks_on()
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", labelsize=12, direction="in", which="both", top=True, right=True)
        ax.set_title(f"{quantity} vs Z (x={x_probe}, y={y_probe})", fontsize=12, pad=10)
        fig.tight_layout()
        out_name = f"analysis_outputs/{prefix}_{quantity}.pdf"
        fig.savefig(out_name, bbox_inches="tight")
        fig.savefig(out_name.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
        print(f"[{prefix}] Saved: {out_name}")
        plt.close(fig)

if __name__ == "__main__":
    cfd_folders = get_plt_folders(base_dir, "plt_N2_", 43500, 50, 6)
    harps_folders = get_plt_folders(base_dir, "plt_harps_N2_", 5800, 20, 30)
    if cfd_folders: generate_z_profile_plots(cfd_folders, cfd_plots_config, prefix="cfd_z_profile")
    if harps_folders: generate_z_profile_plots(harps_folders, harps_plots_config, prefix="harps_z_profile")