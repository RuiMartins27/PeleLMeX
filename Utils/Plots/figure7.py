import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yt

plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})
data_dir = "../../Exec/Production/2DSwirlCase"
SINGLE_ROW = True

panels = [ ("cfd", "temp", "coolwarm", r"$\Delta T_g\ [\mathrm{K}]$"),
           ("cfd", "Y(N)_percent", "PiYG", r"$\Delta Y(N)\ [\%]$") ,
           ("harps", "electron_dens", "PRGn", r"$\Delta n_e\ [\mathrm{m^{-3}}]$"),
           ("harps", "E_over_N", "seismic", r"$\Delta E/N\ [\mathrm{Td}]$"),
           ("cfd", "extsource_rhoh_W_cm3", "RdBu_r", r"$\Delta p_\mathrm{abs}\ [\mathrm{W/cm^3}]$"),]

cases = {   "300W":  {"cfd": "plt_N2_10slm_300W_",  "harps": "plt_harps_N2_10slm_300W_",  "start_step": 56000, "n_steps": 17},
            "600W":  {"cfd": "plt_N2_10slm_600W_",  "harps": "plt_harps_N2_10slm_600W_",  "start_step": 66500, "n_steps": 20},
            "1000W": {"cfd": "plt_N2_10slm_1000W_", "harps": "plt_harps_N2_10slm_1000W_", "start_step": 76000, "n_steps": 23} }

def _extsource_rhoh_W_cm3(field, data):
    return data["boxlib", "extsource_rhoh"] / 1e6

def _Y_N_percent(field, data):
    return data["boxlib", "Y(N)"]*100.0

def register_cfd_fields(ds):
    ds.add_field(name=("gas", "extsource_rhoh_W_cm3"), function=_extsource_rhoh_W_cm3, sampling_type="cell", units="auto", force_override=True)
    ds.add_field(name=("gas", "Y(N)_percent"), function=_Y_N_percent, sampling_type="cell", units="auto", force_override=True)

def compute_series_average(case_cfg, kind, field, slice_axis="theta", register=None):
    prefix, start_step, n_steps = case_cfg[kind], case_cfg["start_step"], case_cfg["n_steps"]
    steps = [start_step - i * 500 for i in range(n_steps)]
    data_sum, bounds, count = None, None, 0
    for s in steps:
        path = os.path.join(data_dir, f"{prefix}{s}")
        if not os.path.exists(path): path = os.path.join(data_dir, f"{prefix}{s:05d}")
        if not os.path.exists(path): continue
        ds = yt.load(path, units_override={"length_unit": (1.0, "m")}); register(ds) if register else None; count += 1
        slc = yt.SlicePlot(ds, slice_axis, field, origin="native"); slc.swap_axes(); frb_data = np.array(slc.frb[field])
        bounds, data_sum = [float(b) for b in slc.frb.bounds] if bounds is None else bounds, np.zeros_like(frb_data, dtype=np.float64) if data_sum is None else data_sum; data_sum += frb_data
    if data_sum is None: raise FileNotFoundError(f"No files found for prefix '{prefix}'")
    return data_sum / max(1, count), bounds

diff_specs = [("1000W", "600W", "1000 W - 600 W")] if SINGLE_ROW else [("600W", "300W", "600 W - 300 W"), ("1000W", "600W", "1000 W - 600 W")]
n_rows = len(diff_specs)
fig, axes = plt.subplots(n_rows, 5, figsize=(16, 5.2 if SINGLE_ROW else 10), sharex=True, sharey=True, squeeze=False)

# Compute averages and generate spatial difference panels
for col, (kind, field, cmap, label) in enumerate(panels):
    register = register_cfd_fields if kind == "cfd" else None
    cache, diffs = {}, []
    for high_k, low_k, row_label in diff_specs:
        if high_k not in cache: cache[high_k], bounds = compute_series_average(cases[high_k], kind, field, register=register)
        if low_k not in cache: cache[low_k], bounds = compute_series_average(cases[low_k], kind, field, register=register)
        diffs.append((cache[high_k] - cache[low_k], row_label))
    bounds_cm = [b * 100 for b in bounds]
    vmax = max(1e-12, max(np.max(np.abs(d[0])) for d in diffs))
    for row, (diff_data, row_label) in enumerate(diffs):
        ax = axes[row, col]
        im = ax.imshow(diff_data, origin="lower", extent=bounds_cm, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        if row == 0: ax.set_title(label, pad=12)
        if row == n_rows - 1: ax.set_xlabel("r [cm]")
        if col == 0: ax.set_ylabel(f"{row_label}\n\nz [cm]" if n_rows > 1 else "z [cm]", fontweight="bold")
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)).ax.tick_params(labelsize=13)

plt.tight_layout(pad=1.2)
out_name = "figure7_power_diff_1row" if SINGLE_ROW else "figure7_power_diff"
plt.savefig(f"{out_name}.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_name}.png", dpi=300, bbox_inches="tight")
plt.close(fig); print(f"Saved {out_name}.pdf")