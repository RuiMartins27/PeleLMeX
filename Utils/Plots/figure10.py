import os
import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data_dir = "../../Exec/Production/2DSwirlCase"
cases = [
    {"label": "5 slm",  "prefix": "plt_N2_5slm_600W_",  "start": 64500, "stride": 500, "n": 20},
    {"label": "10 slm", "prefix": "plt_N2_10slm_600W_", "start": 66500, "stride": 500, "n": 20},
    {"label": "20 slm", "prefix": "plt_N2_20slm_600W_", "start": 71000, "stride": 500, "n": 21},
]

z_radial = [15.0]
r_max = 1.35
n_r_points = 200
r_target = np.linspace(0.0, r_max, n_r_points)

z_min, z_max = 6.0, 26.0
n_z_points = 400
UZ_FACTOR = 1e-0

font_size = 12
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": font_size, "pdf.fonttype": 42})
colors = plt.colormaps["viridis"](np.linspace(0.0, 0.85, len(cases)))

def generate_file_paths(case):
    paths = []
    for i in range(case["n"]):
        idx = case["start"] - i * case["stride"]
        p = os.path.join(data_dir, f"{case['prefix']}{idx}")
        if not os.path.exists(p): p = os.path.join(data_dir, f"{case['prefix']}{idx:05d}")
        if os.path.exists(p): paths.append(p)
    return paths

def _extsource_rhoh_W_cm3(field, data):
    return data["boxlib", "extsource_rhoh"] / 1e6

def register_fields(ds):
    ds.add_field(name=("gas", "extsource_rhoh_W_cm3"), function=_extsource_rhoh_W_cm3, sampling_type="cell", units="auto", force_override=True)

def sample_ray(ds, start, end, coord, fields, target_cm):
    ray = ds.ray(start, end)
    c = ray["index", coord].v * 100.0
    order = np.argsort(c)
    return {f: np.interp(target_cm, c[order], ray[f][order].v) for f in fields}

axial_fields = ["temp", "y_velocity"]
radial_fields = ["temp", ("gas", "extsource_rhoh_W_cm3")]
results = []
z_axis = None

for case in cases:
    files = generate_file_paths(case)
    if not files: raise FileNotFoundError(f"No datasets found for case {case['label']}")
    axial_acc = {f: np.zeros(n_z_points) for f in axial_fields}
    radial_acc = {z0: {f: np.zeros(n_r_points) for f in radial_fields} for z0 in z_radial}
    n_ok = 0
    for path in files:
        try:
            ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
        except Exception:
            continue
        register_fields(ds)
        n_ok += 1
        zlo, zhi = float(ds.domain_left_edge[1].v) * 100.0, float(ds.domain_right_edge[1].v) * 100.0
        eps = 1e-6 * (zhi - zlo)
        z0_ax, z1_ax = max(z_min, zlo + eps), min(z_max, zhi - eps)
        if z_axis is None: z_axis = np.linspace(z0_ax, z1_ax, n_z_points)
        out = sample_ray(ds, [0.0, z0_ax / 100.0, 0.0], [0.0, z1_ax / 100.0, 0.0], "z", axial_fields, z_axis)
        for f in axial_fields: axial_acc[f] += out[f]
        for z0 in z_radial:
            out = sample_ray(ds, [0.0, z0 / 100.0, 0.0], [r_max / 100.0, z0 / 100.0, 0.0], "r", radial_fields, r_target)
            for f in radial_fields: radial_acc[z0][f] += out[f]
    results.append({
        "label": case["label"], "n_ok": n_ok,
        "T_axis": axial_acc["temp"] / n_ok, "uz_axis": axial_acc["y_velocity"] / n_ok * UZ_FACTOR,
        "T_rad": {z0: radial_acc[z0]["temp"] / n_ok for z0 in z_radial},
        "p_rad": {z0: radial_acc[z0][("gas", "extsource_rhoh_W_cm3")] / n_ok for z0 in z_radial},
    })

def twin_legend(ax, left_label, right_label, ls_right="--"):
    handles = [Line2D([0], [0], color=col, lw=1.8, label=res["label"]) for res, col in zip(results, colors)]
    handles += [Line2D([0], [0], color="k", lw=1.5, ls="-", label=left_label), Line2D([0], [0], color="k", lw=1.5, ls=ls_right, label=right_label)]
    ax.legend(handles=handles, loc="best", framealpha=0.85, fontsize=font_size - 1)

z0 = z_radial[0]
fig1, axT2 = plt.subplots(figsize=(6.0, 4.2))
axP = axT2.twinx()
for res, col in zip(results, colors):
    axT2.plot(r_target, res["T_rad"][z0], color=col, lw=1.8, ls="-")
    axP.plot(r_target, res["p_rad"][z0], color=col, lw=1.8, ls="--")
axT2.set_xlabel(r"$r\ [\mathrm{cm}]$")
axT2.set_ylabel(r"$T_g\ [\mathrm{K}]$")
axP.set_ylabel(r"$p_\mathrm{abs}\ [\mathrm{W/cm^3}]$")
axT2.set_xlim(0, r_max)
axT2.grid(True, ls="--", alpha=0.4)
twin_legend(axT2, r"$T_g$ (left)", r"$p_\mathrm{abs}$ (right)")
fig1.tight_layout()
fig1.savefig("figure10_a_flow_rate.png", dpi=300, bbox_inches="tight")
fig1.savefig("figure10_a_flow_rate.pdf", bbox_inches="tight")

fig2, axT = plt.subplots(figsize=(6.5, 4.2))
axU = axT.twinx()
for res, col in zip(results, colors):
    axT.plot(z_axis, res["T_axis"], color=col, lw=1.8, ls="-")
    axU.plot(z_axis, res["uz_axis"], color=col, lw=1.8, ls="--")
axT.set_xlabel(r"$z\ [\mathrm{cm}]$")
axT.set_ylabel(r"$T_g\ [\mathrm{K}]$")
axU.set_ylabel(r"$u_z\ [\mathrm{m/s}]$")
axU.axhline(0.0, color="k", lw=0.6, alpha=0.4)
axT.set_xlim(z_min, z_max)
axT.grid(True, ls="--", alpha=0.4)
twin_legend(axT, r"$T_g$ (left)", r"$u_z$ (right)")
fig2.tight_layout()
fig2.savefig("figure10_b_flow_rate.png", dpi=300, bbox_inches="tight")
fig2.savefig("figure10_b_flow_rate.pdf", bbox_inches="tight")

fig3, (ax1_r, ax2_t) = plt.subplots(1, 2, figsize=(11.0, 4.2))
ax1_p = ax1_r.twinx()
ax2_u = ax2_t.twinx()
for res, col in zip(results, colors):
    ax1_r.plot(r_target, res["T_rad"][z0], color=col, lw=2, ls="-")
    ax1_p.plot(r_target, res["p_rad"][z0], color=col, lw=2, ls="--")
    ax2_t.plot(z_axis, res["T_axis"], color=col, lw=2, ls="-")
    ax2_u.plot(z_axis, res["uz_axis"], color=col, lw=2, ls="--")
ax1_r.set_xlabel(r"$r\ [\mathrm{cm}]$")
ax1_r.set_ylabel(r"$T_g\ [\mathrm{K}]$")
ax1_p.set_ylabel(r"$p_\mathrm{abs}\ [\mathrm{W/cm^3}]$")
ax1_r.set_xlim(0, r_max)
ax1_r.grid(True, ls="--", alpha=0.4)
ax2_t.set_xlabel(r"$z\ [\mathrm{cm}]$")
ax2_t.set_ylabel(r"$T_g\ [\mathrm{K}]$")
ax2_u.set_ylabel(r"$u_z\ [\mathrm{m/s}]$")
ax2_u.axhline(0.0, color="k", lw=0.6, alpha=0.4)
ax2_t.set_xlim(z_min, z_max)
ax2_t.grid(True, ls="--", alpha=0.4)
handles = [Line2D([0], [0], color=col, lw=2, label=res["label"]) for res, col in zip(results, colors)]
handles += [Line2D([0], [0], color="k", lw=2, ls="-", label=r"$T_g$"), Line2D([0], [0], color="k", lw=2, ls="--", label=r"$p_\mathrm{abs}$ or $u_z$")]
fig3.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=len(handles), framealpha=0.85, fontsize=font_size - 1)
fig3.tight_layout()
fig3.savefig("figure10_combined.png", dpi=300, bbox_inches="tight")
fig3.savefig("figure10_combined.pdf", bbox_inches="tight")

out = {"z_axis": z_axis, "r_target": r_target, "z_radial": np.array(z_radial)}
for res in results:
    tag = res["label"].replace(" ", "")
    out[f"T_axis_{tag}"] = res["T_axis"]
    out[f"uz_axis_{tag}"] = res["uz_axis"]
    for zr in z_radial:
        out[f"T_rad_{tag}_z{zr:.3f}"] = res["T_rad"][zr]
        out[f"pabs_rad_{tag}_z{zr:.3f}"] = res["p_rad"][zr]
np.savez_compressed("flow_rate_profiles.npz", **out)
print("Saved flow_rate_profiles.npz")