import os
import yt
import numpy as np
import matplotlib.pyplot as plt

target_quantity = "y_velocity"
plot_rz_map = True
n_z_bins = 200
base_dir = "../../Exec/Production/2DSwirlCase"
series_config = {"base": "plt_N2_600W_no_harps_", "start": 50000, "step": -500}
N_FILES = 20

z_positions = [0.02, 0.06, 0.07, 0.13, 0.20]
r_max = 0.0135
n_r_points = 200
r_target = np.linspace(0.0, r_max, n_r_points)

def generate_file_paths(base_path, series_config, count):
    start_idx, step, prefix = series_config["start"], series_config["step"], series_config["base"]
    indices = [start_idx + i * step for i in range(max(1, count))]
    file_paths = []
    for idx in indices:
        path = os.path.join(base_path, f"{prefix}{idx}")
        if not os.path.exists(path): path = os.path.join(base_path, f"{prefix}{idx:05d}")
        if os.path.exists(path): file_paths.append(path)
        else: print(f"Warning: Dataset not found at {path}")
    return file_paths

files_series = generate_file_paths(base_dir, series_config, N_FILES)
if not files_series: raise FileNotFoundError("No valid datasets found matching the series configuration.")

def register_ang_velocity(ds):
    def _ang_velocity(field, data):
        r_safe = np.maximum(data["index", "r"].v, 1e-12)
        omega = 1e-6 * data["AngMom"].v / (data["density"].v * r_safe)
        return ds.arr(omega, "1/s")
    ds.add_field(name=("gas", "ang_velocity"), function=_ang_velocity, sampling_type="cell", units="1/s", force_override=True)

def fill_empty_radial_bins(arr_2d, used_mask):
    filled = arr_2d.copy()
    n_r, n_z = filled.shape
    r_idx = np.arange(n_r)
    for j in range(n_z):
        valid = used_mask[:, j] & (filled[:, j] != 0) & ~np.isnan(filled[:, j])
        if np.any(valid) and not np.all(valid): filled[:, j] = np.interp(r_idx, r_idx[valid], filled[valid, j])
    return filled

is_ang_mom = (target_quantity == "AngMom")
quantities_to_process = [target_quantity, "ang_velocity"] if is_ang_mom else [target_quantity]

accumulators = {q: {z: np.zeros_like(r_target) for z in z_positions} for q in quantities_to_process}
accumulators_2d = {q: None for q in quantities_to_process}
r_bins_2d, z_bins_2d, used_2d_mask = None, None, None
successful_files = 0

for file_idx, path in enumerate(files_series):
    print(f"[{file_idx + 1}/{len(files_series)}] Loading: {path}")
    try: ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
    except Exception as e:
        print(f"  Skipping {path}: {e}")
        continue
    if is_ang_mom: register_ang_velocity(ds)
    successful_files += 1

    for z0 in z_positions:
        ray = ds.ray([0.0, z0, 0.0], [r_max, z0, 0.0])
        sort_idx = ray["index", "r"].argsort()
        r_coords = ray["index", "r"][sort_idx].v
        for q in quantities_to_process:
            field_key = ("gas", "ang_velocity") if q == "ang_velocity" else q
            vals = ray[field_key][sort_idx].v
            accumulators[q][z0] += np.interp(r_target, r_coords, vals)

    if plot_rz_map:
        yt_fields = [("gas", "ang_velocity") if q == "ang_velocity" else q for q in quantities_to_process]
        prof2d = yt.create_profile(data_source=ds.all_data(), bin_fields=[("index", "r"), ("index", "z")], fields=yt_fields, weight_field="cell_volume", n_bins=[n_r_points, n_z_bins])
        if r_bins_2d is None:
            r_bins_2d, z_bins_2d = prof2d.x.v, prof2d.y.v
            used_2d_mask = np.array(prof2d.used, dtype=bool)
        else: used_2d_mask |= np.array(prof2d.used, dtype=bool)
        for q, f_key in zip(quantities_to_process, yt_fields):
            data2d = prof2d[f_key].v
            if accumulators_2d[q] is None: accumulators_2d[q] = np.zeros_like(data2d, dtype=np.float64)
            accumulators_2d[q] += data2d

if successful_files == 0: raise RuntimeError("No datasets were successfully loaded and processed.")

cmap = plt.colormaps["viridis"]
num_lines = len(z_positions)

for q in quantities_to_process:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, z0 in enumerate(z_positions):
        avg_vals = accumulators[q][z0] / successful_files
        ax.plot(r_target, avg_vals, color=cmap(i / (num_lines - 1)) if num_lines > 1 else cmap(0.0), linewidth=2, label=f"$z = {z0}$")
    ax.set_xlabel("Radius $r$ (m)", fontsize=12, family="serif")
    if q == "ang_velocity":
        ax.set_ylabel(r"Azimuthal Velocity ($m/s$)", fontsize=12, family="serif")
        ax.set_title(f"Time-Averaged Azimuthal Velocity ($N = {successful_files}$ Datasets)", fontsize=13, family="serif", weight="bold")
    elif q == "AngMom":
        ax.set_ylabel("Angular Momentum (native units)", fontsize=12, family="serif")
        ax.set_title(f"Time-Averaged Angular Momentum ($N = {successful_files}$ Datasets)", fontsize=13, family="serif", weight="bold")
    else:
        ax.set_ylabel(f"{q} (native units)", fontsize=12, family="serif")
        ax.set_title(f"Time-Averaged {q} ($N = {successful_files}$ Datasets)", fontsize=13, family="serif", weight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="best", framealpha=0.8)
    ax.set_xlim(0, r_max)
    out_file = f"1d_{q}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved {out_file}")
    plt.close(fig)

output_data_2d = {"z_positions": np.array(z_positions), "successful_files": successful_files}
for q in quantities_to_process:
    for z0 in z_positions:
        z_key = f"{z0:.3f}"
        output_data_2d[f"r_{z_key}"] = r_target
        output_data_2d[f"{q}_{z_key}"] = accumulators[q][z0] / successful_files
np.savez_compressed("profile_data_2d.npz", **output_data_2d)
print("Saved 2D profile data to 'profile_data_2d.npz'")

if plot_rz_map:
    rz_output_data = {"r_bins": r_bins_2d, "z_bins": z_bins_2d, "successful_files": successful_files}
    for q in quantities_to_process:
        avg_2d = accumulators_2d[q] / successful_files
        avg_2d = fill_empty_radial_bins(avg_2d, used_2d_mask)
        rz_output_data[q] = avg_2d
        fig, ax = plt.subplots(figsize=(8, 6))
        R, Z = np.meshgrid(r_bins_2d, z_bins_2d, indexing='ij')
        c = ax.pcolormesh(R, Z, avg_2d, shading='nearest', cmap='viridis')
        fig.colorbar(c, ax=ax, label=q)
        ax.set_xlabel("Radius $r$ (m)", fontsize=12, family="serif")
        ax.set_ylabel("z (m)", fontsize=12, family="serif")
        ax.set_title(f"Time-Averaged {q} R-Z Map", fontsize=13, family="serif", weight="bold")
        out_file = f"rz_map_{q}.png"
        #fig.savefig(out_file, dpi=300, bbox_inches="tight")
        #print(f"Saved {out_file}")
        plt.close(fig)
    np.savez_compressed("rz_map_data.npz", **rz_output_data)
    print("Saved 2D R-Z map data to 'rz_map_data.npz'")