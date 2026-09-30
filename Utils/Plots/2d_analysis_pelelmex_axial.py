import os
import yt
import numpy as np
import matplotlib.pyplot as plt


# Target quantity can be "AngMom", "temp", "density", "y_velocity", "x_velocity"
target_quantity = "AngMom"
base_dir = "../../Exec/Production/2DSwirlCase"
series_config = {"base": "plt_N2_600W_no_harps_", "start": 50000, "step": -500}
N_FILES = 20

r_max = 0.013
n_r_points = 200
r_target = np.linspace(0.0, r_max, n_r_points)

z_min = 0.0
z_max = 0.31
n_z_points = 100
z_scan = np.linspace(z_min, z_max, n_z_points)


def generate_file_paths(base_path, series_config, count):
    start_idx = series_config["start"]
    step = series_config["step"]
    prefix = series_config["base"]

    indices = [start_idx + i * step for i in range(max(1, count))]

    file_paths = []
    for idx in indices:
        path = os.path.join(base_path, f"{prefix}{idx}")
        if not os.path.exists(path):
            path = os.path.join(base_path, f"{prefix}{idx:05d}")
        if os.path.exists(path):
            file_paths.append(path)
        else:
            print(f"Warning: Dataset not found at {path}")

    return file_paths

files_series = generate_file_paths(base_dir, series_config, N_FILES)

if not files_series:
    raise FileNotFoundError("No valid datasets found matching the series configuration.")


def register_ang_velocity(ds):
    def _ang_velocity(field, data):
        r = data["index", "r"].v
        r_safe = np.maximum(r, 1e-12)

        ang_mom = data["AngMom"].v
        rho = data["density"].v

        omega = 1e-6 * ang_mom / (rho * r_safe)
        return ds.arr(omega, "1/s")

    ds.add_field(
        name=("gas", "ang_velocity"),
        function=_ang_velocity,
        sampling_type="cell",
        units="1/s",
        force_override=True
    )


def area_weighted_average(vals, r):
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    numerator = trapz_fn(vals * r, r)
    return 2.0 * numerator / (r[-1] ** 2)


is_ang_mom = (target_quantity == "AngMom")

quantities_to_process = [target_quantity]
if is_ang_mom:
    quantities_to_process.append("ang_velocity")

# One accumulated scalar per z position (area-averaged over r), summed over files
accumulators = {q: np.zeros_like(z_scan) for q in quantities_to_process}

successful_files = 0

for file_idx, path in enumerate(files_series):
    print(f"[{file_idx + 1}/{len(files_series)}] Loading: {path}")
    try:
        ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
    except Exception as e:
        print(f"  Skipping {path}: {e}")
        continue

    if is_ang_mom:
        register_ang_velocity(ds)

    successful_files += 1

    for z_idx, z0 in enumerate(z_scan):
        ray_start = [0.0, z0, 0.0]
        ray_end = [r_max, z0, 0.0]

        ray = ds.ray(ray_start, ray_end)

        sort_idx = ray["index", "r"].argsort()
        r_coords = ray["index", "r"][sort_idx].v

        for q in quantities_to_process:
            field_key = ("gas", "ang_velocity") if q == "ang_velocity" else q
            vals = ray[field_key][sort_idx].v
            interp_vals = np.interp(r_target, r_coords, vals)

            accumulators[q][z_idx] += area_weighted_average(interp_vals, r_target)

if successful_files == 0:
    raise RuntimeError("No datasets were successfully loaded and processed.")

# -----------------------------------------------------------------------------
# Plot Time- & Area-Averaged Profiles vs z (single curve per quantity)
# -----------------------------------------------------------------------------
for q in quantities_to_process:
    avg_vals = accumulators[q] / successful_files

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(z_scan, avg_vals, color="navy", linewidth=2)

    ax.set_xlabel("Axial position $z$ (m)", fontsize=12, family="serif")

    if q == "ang_velocity":
        ax.set_ylabel(r"Area-Averaged Azimuthal Velocity ($m/s$)", fontsize=12, family="serif")
        ax.set_title(f"Time- & Area-Averaged Azimuthal Velocity ($N = {successful_files}$ Datasets)",
                     fontsize=13, family="serif", weight="bold")
    elif q == "AngMom":
        ax.set_ylabel("Area-Averaged Angular Momentum (native units)", fontsize=12, family="serif")
        ax.set_title(f"Time- & Area-Averaged Angular Momentum ($N = {successful_files}$ Datasets)",
                     fontsize=13, family="serif", weight="bold")
    else:
        ax.set_ylabel(f"Area-Averaged {q} (native units)", fontsize=12, family="serif")
        ax.set_title(f"Time- & Area-Averaged {q} ($N = {successful_files}$ Datasets)",
                     fontsize=13, family="serif", weight="bold")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(z_min, z_max)

    out_file = f"1d_axial_{q}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved {out_file}")
    plt.close(fig)