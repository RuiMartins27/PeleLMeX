import os, yt, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.titlesize": 13,
    "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 13, "lines.linewidth": 2.5, "lines.markersize": 6
})

data_dir = "../../Exec/Production/2DSwirlCase"

cases_harps = {
    300: {"cfd": "plt_N2_10slm_300W_", "start_step": 56000, "n_steps": 17},
    450: {"cfd": "plt_N2_10slm_450W_", "start_step": 60000, "n_steps": 17},
    600: {"cfd": "plt_N2_10slm_600W_", "start_step": 66500, "n_steps": 20},
    800: {"cfd": "plt_N2_10slm_800W_", "start_step": 70000, "n_steps": 20},
    1000: {"cfd": "plt_N2_10slm_1000W_", "start_step": 76000, "n_steps": 23},
    1250: {"cfd": "plt_N2_10slm_1250W_", "start_step": 80000, "n_steps": 23},
    1500: {"cfd": "plt_N2_10slm_1500W_", "start_step": 73500, "n_steps": 20}
}

cases_static = {
    300: {"cfd": "plt_N2_10slm_300W_static_", "start_step": 62000, "n_steps": 17},
    450: {"cfd": "plt_N2_10slm_450W_static_", "start_step": 74500, "n_steps": 20},
    600: {"cfd": "plt_N2_10slm_600W_static_", "start_step": 85000, "n_steps": 22},
    800: {"cfd": "plt_N2_10slm_800W_static_", "start_step": 106500, "n_steps": 24},
    1000: {"cfd": "plt_N2_10slm_1000W_static_", "start_step": 106500, "n_steps": 23},
    1250: {"cfd": "plt_N2_10slm_1250W_static_", "start_step": 106500, "n_steps": 24},
    1500: {"cfd": "plt_N2_10slm_1500W_static_", "start_step": 106500, "n_steps": 22}
}

cases_cylinder = {
    300: {"cfd": "plt_N2_10slm_300W_cylinder_", "start_step": 43000, "n_steps": 14},
    450: {"cfd": "plt_N2_10slm_450W_cylinder_", "start_step": 56000, "n_steps": 17},
    600: {"cfd": "plt_N2_10slm_600W_cylinder_", "start_step": 64500, "n_steps": 20},
    800: {"cfd": "plt_N2_10slm_800W_cylinder_", "start_step": 73000, "n_steps": 23},
    1000: {"cfd": "plt_N2_10slm_1000W_cylinder_", "start_step": 89000, "n_steps": 26},
    1250: {"cfd": "plt_N2_10slm_1250W_cylinder_", "start_step": 95000, "n_steps": 28},
    1500: {"cfd": "plt_N2_10slm_1500W_cylinder_", "start_step": 100000, "n_steps": 30}
}

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
        path = os.path.join(data_dir, f"{prefix}{s:05d}") if not os.path.exists(path) else path
        if not os.path.exists(path): continue
        ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
        register(ds) if register else None
        count += 1
        slc = yt.SlicePlot(ds, slice_axis, field, origin="native")
        slc.swap_axes()
        frb_data = np.array(slc.frb[field])
        bounds = [float(b) for b in slc.frb.bounds] if bounds is None else bounds
        data_sum = np.zeros_like(frb_data, dtype=np.float64) if data_sum is None else data_sum
        data_sum += frb_data
    if data_sum is None:
        raise FileNotFoundError(f"No files found for prefix '{prefix}'")
    return data_sum / max(1, count), bounds

def extract_metrics(case_dict):
    powers, peak_T, core_p_W, centroid_r_p, half_r = [], [], [], [], []
    for p in sorted(case_dict.keys()):
        try:
            T_avg, bT = compute_series_average(case_dict[p], "cfd", "temp")
            P_avg, bP = compute_series_average(case_dict[p], "cfd", "extsource_rhoh_W_cm3", register=register_cfd_fields)
        except FileNotFoundError:
            continue
        r_T, z_T = np.linspace(bT[0], bT[1], T_avg.shape[1]), np.linspace(bT[2], bT[3], T_avg.shape[0])
        r_P, z_P = np.linspace(bP[0], bP[1], P_avg.shape[1]), np.linspace(bP[2], bP[3], P_avg.shape[0])
        R_T, R_P = np.meshgrid(r_T, z_T)[0], np.meshgrid(r_P, z_P)[0]
        peak_T.append(np.max(T_avg))
        centroid_r_p.append((np.sum(R_P**2 * P_avg) / np.sum(R_P * P_avg)) * 100)
        rm, zm = (r_P >= 0.0) & (r_P <= 0.004), (z_P >= 0.14) & (z_P <= 0.15)
        dr, dz = r_P[1] - r_P[0], z_P[1] - z_P[0]
        p_int_W = np.sum(P_avg[np.ix_(zm, rm)] * 1e6 * 2 * np.pi * R_P[np.ix_(zm, rm)] * dr * dz)
        core_p_W.append(p_int_W)
        zi, ri = np.unravel_index(np.argmax(T_avg), T_avg.shape)
        T_slice, T_half = T_avg[zi, :], T_avg[zi, ri] / 2.0
        idx = np.where(T_slice < T_half)[0]
        if len(idx) > 0 and idx[0] > 0:
            half_r.append((r_T[idx[0]-1] + (r_T[idx[0]] - r_T[idx[0]-1]) * (T_half - T_slice[idx[0]-1]) / (T_slice[idx[0]] - T_slice[idx[0]-1])) * 100)
        else:
            half_r.append((r_T[0] if len(idx) > 0 else r_T[-1]) * 100)
        powers.append(p)
    return powers, peak_T, core_p_W, centroid_r_p, half_r

pw_harps, T_harps, P_harps, rp_harps, rThalf_harps = extract_metrics(cases_harps)
pw_stat, T_stat, P_stat, rp_stat, rThalf_stat = extract_metrics(cases_static)
pw_cyl, T_cyl, P_cyl, rp_cyl, rThalf_cyl = extract_metrics(cases_cylinder)

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
c_blue, c_red = "#0055A5", "#C8102E"

ax2 = ax1.twinx()
l1 = ax1.plot(pw_harps, P_harps, 'o-', color=c_blue, label=r"$P_{\mathrm{abs,core}}$ (EM solver)")
l2 = ax2.plot(pw_harps, T_harps, 's-', color=c_red, label=r"$T_{\mathrm{g,peak}}$ (EM solver)")
lines1, labels1 = l1 + l2, [l.get_label() for l in l1 + l2]
if pw_stat:
    l3 = ax1.plot(pw_stat, P_stat, 'o--', color=c_blue, alpha=0.8, label=r"$P_{\mathrm{abs,core}}$ (static)")
    l4 = ax2.plot(pw_stat, T_stat, 's--', color=c_red, alpha=0.8, label=r"$T_{\mathrm{g,peak}}$ (static)")
    lines1 += l3 + l4
    labels1 += [l.get_label() for l in l3 + l4]
if pw_cyl:
    l9 = ax1.plot(pw_cyl, P_cyl, 'o:', color=c_blue, alpha=0.8, label=r"$P_{\mathrm{abs,core}}$ (cylinder)")
    l10 = ax2.plot(pw_cyl, T_cyl, 's:', color=c_red, alpha=0.8, label=r"$T_{\mathrm{g,peak}}$ (cylinder)")
    lines1 += l9 + l10
    labels1 += [l.get_label() for l in l9 + l10]

ax1.set_xlabel(r"$P_{\mathrm{in}}$ [W]")
ax1.set_ylabel(r"$P_{\mathrm{abs,core}}$ [W]", color=c_blue)
ax1.tick_params(axis='y', labelcolor=c_blue, direction="in", which="both")
ax1.tick_params(axis='x', direction="in", which="both")
ax2.set_ylabel(r"$T_{\mathrm{g,peak}}$ [K]", color=c_red)
ax2.tick_params(axis='y', labelcolor=c_red, direction="in", which="both")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

ax4 = ax3.twinx()
l5 = ax3.plot(pw_harps, rp_harps, 'o-', color=c_blue, label=r"$\bar{r}_{\mathrm{p}}$ (EM solver)")
l6 = ax4.plot(pw_harps, rThalf_harps, 's-', color=c_red, label=r"$r_{\mathrm{T_{g},half}}$ (EM solver)")
lines2, labels2 = l5 + l6, [l.get_label() for l in l5 + l6]
if pw_stat:
    l7 = ax3.plot(pw_stat, rp_stat, 'o--', color=c_blue, alpha=0.8, label=r"$\bar{r}_{\mathrm{p}}$ (static)")
    l8 = ax4.plot(pw_stat, rThalf_stat, 's--', color=c_red, alpha=0.8, label=r"$r_{\mathrm{T_{g},half}}$ (static)")
    lines2 += l7 + l8
    labels2 += [l.get_label() for l in l7 + l8]
if pw_cyl:
    l11 = ax3.plot(pw_cyl, rp_cyl, 'o:', color=c_blue, alpha=0.8, label=r"$\bar{r}_{\mathrm{p}}$ (cylinder)")
    l12 = ax4.plot(pw_cyl, rThalf_cyl, 's:', color=c_red, alpha=0.8, label=r"$r_{\mathrm{T_{g},half}}$ (cylinder)")
    lines2 += l11 + l12
    labels2 += [l.get_label() for l in l11 + l12]

ax3.set_xlabel(r"$P_{\mathrm{in}}$ [W]")
ax3.set_ylabel(r"$r_{\mathrm{p}}$ [cm]", color=c_blue)
ax3.tick_params(axis='y', labelcolor=c_blue, direction="in", which="both")
ax3.tick_params(axis='x', direction="in", which="both")
ax4.set_ylabel(r"$r_{\mathrm{T,half}}$ [cm]", color=c_red)
ax4.tick_params(axis='y', labelcolor=c_red, direction="in", which="both")
ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax3.set_ylim(0, 1.15)
ax4.set_ylim(0, 1.15)

legend_handles = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=3, label="EM solver"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=3, label="Parabolic"),
    Line2D([0], [0], color="black", linestyle=":", linewidth=3, label="Cylinder")
]

fig.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.065),
    ncol=3, frameon=True, framealpha=0.6, edgecolor="grey", handlelength=3.5
)

plt.tight_layout()
plt.savefig("figure8_metrics.pdf", bbox_inches="tight")
plt.savefig("figure8_metrics.png", dpi=300, bbox_inches="tight")