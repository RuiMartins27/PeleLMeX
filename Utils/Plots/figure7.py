import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yt

plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})
data_dir = "../../Exec/Production/2DSwirlCase"

panels = [  ("cfd", "temp", "coolwarm", r"$\Delta T_g\ [\mathrm{K}]$"),
            ("harps", "electron_dens", "PRGn", r"$\Delta n_e\ [\mathrm{m^{-3}}]$"),
            ("harps", "E_over_N", "seismic", r"$\Delta E/N\ [\mathrm{Td}]$"),
            ("cfd", "extsource_rhoh_W_cm3", "RdBu_r", r"$\Delta p_\mathrm{abs}\ [\mathrm{W/cm^3}]$")]

cases = {   "300W":  {"cfd": "plt_N2_10slm_300W_",  "harps": "plt_harps_N2_10slm_300W_",  "start_step": 56000, "n_steps": 17},
            "600W":  {"cfd": "plt_N2_10slm_600W_",  "harps": "plt_harps_N2_10slm_600W_",  "start_step": 66500, "n_steps": 20},
            "1000W": {"cfd": "plt_N2_10slm_1000W_", "harps": "plt_harps_N2_10slm_1000W_", "start_step": 76000, "n_steps": 23}}

def _extsource_rhoh_W_cm3(field, data):
    return data["boxlib", "extsource_rhoh"] / 1e6

def register_cfd_fields(ds):
    ds.add_field(name=("gas", "extsource_rhoh_W_cm3"), function=_extsource_rhoh_W_cm3, sampling_type="cell", units="auto", force_override=True)

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

fig, axes = plt.subplots(2, 4, figsize=(14, 10), sharex=True, sharey=True)
for col, (kind, field, cmap, label) in enumerate(panels):
    register = register_cfd_fields if kind == "cfd" else None
    middle_case, bounds = compute_series_average(cases["600W"], kind, field, register=register)
    lower_case, _ = compute_series_average(cases["300W"], kind, field, register=register)
    higher_case, _ = compute_series_average(cases["1000W"], kind, field, register=register)
    bounds_cm = [b * 100 for b in bounds]
    diffs = [(middle_case - lower_case, "600 W - 300 W"), (higher_case - middle_case, "1000 W - 600 W")]
    vmax = max(1e-12, np.max(np.abs(diffs[0][0])), np.max(np.abs(diffs[1][0])))
    for row, (diff_data, row_label) in enumerate(diffs):
        ax = axes[row, col]; im = ax.imshow(diff_data, origin="lower", extent=bounds_cm, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        if row == 0: ax.set_title(label, pad=12)
        if row == 1: ax.set_xlabel("r [cm]")
        if col == 0: ax.set_ylabel(f"{row_label}\n\nz [cm]", fontweight="bold")
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)).ax.tick_params(labelsize=13)

plt.tight_layout(pad=1.2)
plt.savefig("figure7_power_diff.pdf", dpi=300, bbox_inches="tight")
plt.close(fig); print("Saved figure7_power_diff.pdf")