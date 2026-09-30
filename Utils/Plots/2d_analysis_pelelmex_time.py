import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yt

base_dir = "../../Exec/Production/2DSwirlCase"
file_prefix = "plt_N2_10slm_600W_"
use_all_plt = True
base_step = 66500
step_size = 500
n_folders = 200

r_probe = 0.000
z_probe = None
z_min = 0.10
z_max = 0.20
n_z_points = 100

quantities = [
    ("temp", "Probe Temperature (K)", "tab:red", "o"),
    ("y_velocity", "Probe Axial Velocity (m/s)", "tab:blue", "s"),
]

trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def get_all_plt_dirs(base_dir):
    plt_dirs = [p for p in Path(base_dir).glob(f"{file_prefix}*") if p.is_dir()]
    def extract_step(path):
        match = re.search(rf"{file_prefix}(\d+)$", path.name)
        return int(match.group(1)) if match else -1
    return sorted([str(p) for p in plt_dirs if extract_step(p) != -1], key=lambda p: extract_step(Path(p)))

def parse_header_time(plt_dir):
    with open(os.path.join(plt_dir, "Header"), "r") as f:
        lines = [line.strip() for line in f]
    return float(lines[3 + int(lines[1])])

def probe_value(ds, field):
    if z_probe is not None:
        return float(ds.point([r_probe, z_probe, 0.0])[field].v)
    ray = ds.ray([r_probe, z_min, 0.0], [r_probe, z_max, 0.0])
    sort_idx = np.argsort(ray["index", "z"].v)
    z_coords = ray["index", "z"][sort_idx].v
    z_grid = np.linspace(z_min, z_max, n_z_points)
    return float(trapz_fn(np.interp(z_grid, z_coords, ray[field][sort_idx].v), z_grid) / (z_max - z_min))

if use_all_plt:
    plt_folders = get_all_plt_dirs(base_dir)
else:
    plt_folders = []
    for i in range(n_folders):
        step = base_step - i * step_size
        path = f"{base_dir}/{file_prefix}{step}" if os.path.exists(f"{base_dir}/{file_prefix}{step}") else f"{base_dir}/{file_prefix}{step:05d}"
        if os.path.exists(path): plt_folders.append(path)

if not plt_folders:
    raise FileNotFoundError(f"No '{file_prefix}*' datasets found under {base_dir}")
print(f"Found {len(plt_folders)} datasets.")

times, data = [], {field: [] for field, _, _, _ in quantities}
# Extract values across all time steps
for i, path in enumerate(plt_folders):
    print(f"[{i + 1}/{len(plt_folders)}] {path}")
    try:
        t = parse_header_time(path)
        ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
        times.append(t)
        for field, _, _, _ in quantities:
            data[field].append(probe_value(ds, field))
    except Exception as e:
        print(f"  Skipping {path}: {e}")

if not times:
    raise RuntimeError("No datasets were successfully processed.")

times = np.array(times)
sort_idx = np.argsort(times)
times = times[sort_idx]
for field in data:
    data[field] = np.array(data[field])[sort_idx]

(field1, label1, color1, marker1), (field2, label2, color2, marker2) = quantities

# Publication styling defaults
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'mathtext.fontset': 'stix'})

fig, ax1 = plt.subplots(figsize=(7, 4.5))

# Primary Axis (Temperature)
ax1.plot(times, data[field1], color=color1, linewidth=1.5, marker=marker1, markersize=3, label="Temperature")
ax1.set_xlabel("Time (s)", fontsize=14, labelpad=6)
ax1.set_ylabel(label1, fontsize=14, color=color1, labelpad=6)
ax1.tick_params(axis="y", labelcolor=color1, labelsize=12, direction="in", which="both", top=True)
ax1.tick_params(axis="x", labelsize=12, direction="in", which="both", top=True)
ax1.minorticks_on()
ax1.grid(True, linestyle=":", alpha=0.6)

# Secondary Axis (Velocity)
ax2 = ax1.twinx()
ax2.plot(times, data[field2], color=color2, linewidth=1.5, marker=marker2, markersize=3, label="Axial Velocity")
ax2.set_ylabel(label2, fontsize=14, color=color2, labelpad=6)
ax2.tick_params(axis="y", labelcolor=color2, labelsize=12, direction="in", which="both")
ax2.minorticks_on()

# Clean up layout and save vector + raster formats
probe_desc = f"r={r_probe}, z={z_probe} m" if z_probe is not None else f"r={r_probe}, line avg (z={z_min}-{z_max} m)"
ax1.set_title(f"Temperature & Axial Velocity vs. Time ({probe_desc})", fontsize=13, pad=10)

fig.tight_layout()
fig.savefig("time_temp_vel.pdf", bbox_inches="tight")
fig.savefig("time_temp_vel.png", dpi=600, bbox_inches="tight")
print("Saved time_temp_vel.pdf and time_temp_vel.png (600 DPI)")
plt.close(fig)