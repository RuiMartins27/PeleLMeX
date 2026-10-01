import os
import yt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, MultipleLocator

data_dir     = "../../Exec/Production/2DSwirlCase"
cfd_prefix   = "plt_N2_10slm_600W_corr_"
harps_prefix = "plt_harps_N2_10slm_600W_corr_"

start_step  = 12500
step_stride = 500
n_datasets  = 3

use_cache = True
out_pdf   = "figure6_ss_panel.pdf"
out_png   = "figure6_ss_panel.png"

steps = [start_step - i * step_stride for i in range(n_datasets)]

panels = [
    ("cfd",   "temp",                 "magma",   r"$T_g\ [\mathrm{K}]$",             None),
    ("cfd",   "ang_velocity",         "CMRmap",  r"$u_\theta\ [\mathrm{m/s}]$",      None),
    ("cfd",   "y_velocity",           "RdBu_r",  r"$u_z\ [\mathrm{m/s}]$",           (-5, 5)),
    ("harps", "electron_dens",        "viridis", r"$n_e\ [\mathrm{m^{-3}}]$",        None),
    ("harps", "E_over_N",             "plasma",  r"$E/N\ [\mathrm{Td}]$",            None),
    ("cfd",   "extsource_rhoh_W_cm3", "inferno", r"$p_\mathrm{abs}\ [\mathrm{W/cm^3}]$", None),
]

fig_w       = 7.5
left_in     = 0.60
cbar_pad    = 0.08
cbar_w      = 0.25
right_in    = 0.85
gap_in      = 0.40
top_in      = 0.30
bottom_in   = 0.60
x_tick_step = 5.0
r_tick_step = 1.0
font_size   = 9
interpolation = "nearest"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": font_size,
    "pdf.fonttype": 42,
})

def _ang_velocity(field, data):
    r_safe = np.maximum(data["index", "r"], data.ds.quan(1e-12, "m"))
    return 1e-4 * data["AngMom"] / (data["density"] * r_safe)

def _extsource_rhoh_W_cm3(field, data):
    return data["boxlib", "extsource_rhoh"] / 1e6

def register_cfd_fields(ds):
    ds.add_field(name=("gas", "ang_velocity"), function=_ang_velocity, sampling_type="cell", units="cm**2/g", force_override=True)
    ds.add_field(name=("gas", "extsource_rhoh_W_cm3"), function=_extsource_rhoh_W_cm3, sampling_type="cell", units="auto", force_override=True)

def load_series(prefix, register=None):
    dss = []
    for s in steps:
        path = os.path.join(data_dir, f"{prefix}{s}")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
        if register is not None:
            register(ds)
        dss.append(ds)
    print(f"Loaded {len(dss)}/{len(steps)} datasets for prefix '{prefix}'")
    return dss

def average_fields(dss, names):
    level = max(ds.index.max_level for ds in dss)
    sums = {}
    for i, ds in enumerate(dss, start=1):
        print(f"  [{i}/{len(dss)}] {ds}")
        dims = ds.domain_dimensions.copy()
        dims[: ds.dimensionality] *= ds.relative_refinement(0, level)
        cg = ds.covering_grid(level, left_edge=ds.domain_left_edge, dims=dims)
        for name in names:
            arr = cg[name]
            if name not in sums:
                sums[name] = (np.zeros(arr.shape, dtype=np.float64), str(arr.units))
            sums[name][0][...] += arr.to(sums[name][1]).d
    return {n: s / len(dss) for n, (s, _) in sums.items()}

def get_average(kind, prefix, fields, register=None):
    dss = load_series(prefix, register)
    if not dss:
        raise SystemExit(f"No {kind} snapshots found in {data_dir}")
    cache = f"avg_panel_{kind}_{start_step}_{step_stride}_{n_datasets}.npz"
    if use_cache and os.path.exists(cache):
        cached = dict(np.load(cache))
        if all(f in cached for f in fields):
            print(f"Using cached averages: {cache}")
            return {f: cached[f] for f in fields}, dss[0]
    print(f"Averaging {kind} fields...")
    arrays = average_fields(dss, fields)
    np.savez(cache, **arrays)
    return arrays, dss[0]

def to_rz_image(arr, axis_order):
    order = list(axis_order)
    i_r, i_z, i_t = order.index("r"), order.index("z"), order.index("theta")
    img = np.take(arr, 0, axis=i_t)
    return img if i_r < i_z else img.T

cfd_fields   = [p[1] for p in panels if p[0] == "cfd"]
harps_fields = [p[1] for p in panels if p[0] == "harps"]

cfd_avg,   ds_cfd   = get_average("cfd",   cfd_prefix,   cfd_fields,   register_cfd_fields)
harps_avg, ds_harps = get_average("harps", harps_prefix, harps_fields) if harps_fields else ({}, None)

axis_order = list(ds_cfd.coordinates.axis_order)
le = ds_cfd.domain_left_edge.to("cm").d
re = ds_cfd.domain_right_edge.to("cm").d
zmin, zmax = le[axis_order.index("z")], re[axis_order.index("z")]
rmin, rmax = le[axis_order.index("r")], re[axis_order.index("r")]

n = len(panels)
ax_w    = fig_w - left_in - cbar_pad - cbar_w - right_in
panel_h = ax_w * (rmax - rmin) / (zmax - zmin)
fig_h   = top_in + n * panel_h + (n - 1) * gap_in + bottom_in

fig = plt.figure(figsize=(fig_w, fig_h))
axes = []
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for i, (source, field, cmap, label, limits) in enumerate(panels):
    arr3d = cfd_avg[field] if source == "cfd" else harps_avg[field]
    img = to_rz_image(arr3d, axis_order)

    vmin, vmax = limits if limits is not None else (np.nanmin(img), np.nanmax(img))

    y0 = bottom_in + (n - 1 - i) * (panel_h + gap_in)
    ax = fig.add_axes([left_in / fig_w, y0 / fig_h, ax_w / fig_w, panel_h / fig_h], sharex=axes[0] if axes else None)
    cax = fig.add_axes([(left_in + ax_w + cbar_pad) / fig_w, y0 / fig_h, cbar_w / fig_w, panel_h / fig_h])
    axes.append(ax)

    im = ax.imshow(img, origin="lower", extent=[zmin, zmax, rmin, rmax], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)

    cb = fig.colorbar(im, cax=cax)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    cb.set_label(label, labelpad=3)

    ax.text(0.02, 1.66, panel_labels[i], transform=ax.transAxes, ha="left", va="top", fontsize=font_size + 1, fontweight="bold", color="black")

    ax.yaxis.set_major_locator(MultipleLocator(r_tick_step))
    ax.tick_params(direction="in", top=True, right=True)
    cb.ax.tick_params(direction="in")
    if i < n - 1:
        ax.tick_params(labelbottom=False)

axes[-1].set_xlim(zmin, zmax)
axes[-1].xaxis.set_major_locator(MultipleLocator(x_tick_step))
axes[-1].set_xlabel(r"$z\ (\mathrm{cm})$")

# Centered single y-axis label spanning all subplots
y_mid = (bottom_in + 0.5 * (n * panel_h + (n - 1) * gap_in)) / fig_h
fig.text(0.04, y_mid, r"$r\ (\mathrm{cm})$", va="center", ha="center", rotation="vertical")

fig.savefig(out_pdf, format="pdf")
fig.savefig(out_png, dpi=900, format="png")
print(f"Saved: {out_pdf}")