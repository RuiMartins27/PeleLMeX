import os
import numpy as np
import yt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator

plt.rcParams.update({
    "font.family": "serif", "font.size": 13.5, "axes.titlesize": 13.5,
    "axes.labelsize": 13.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 13, "lines.linewidth": 2.5, "lines.markersize": 7,
})

data_dir = "../../Exec/Production/2DSwirlCase"
Z_TARGET = 0.198
SNAP_DT = 500
FLOW_IN_SLM = 10.0
P_ATM, R_GAS = 101325.0, 8.314462618
M_N, M_N2 = 14.0067e-3, 28.0134e-3
V_MOL = 22414.0
R_TUBE, T_MIN = 0.0135, 200.0
BLUE, RED = "#0055A5", "#C8102E"

COST_MIN, COST_MAX = 0.47, 220      # energy-cost window for the z-profile figure

cases_harps = {
    300: {"cfd": "plt_N2_10slm_300W_", "start_step": 56000, "n_steps": 17},
    450: {"cfd": "plt_N2_10slm_450W_", "start_step": 60000, "n_steps": 17},
    600: {"cfd": "plt_N2_10slm_600W_", "start_step": 66500, "n_steps": 20},
    800: {"cfd": "plt_N2_10slm_800W_", "start_step": 70000, "n_steps": 20},
    1000: {"cfd": "plt_N2_10slm_1000W_", "start_step": 76000, "n_steps": 23},
    1250: {"cfd": "plt_N2_10slm_1250W_", "start_step": 80000, "n_steps": 23},
    1500: {"cfd": "plt_N2_10slm_1500W_", "start_step": 73500, "n_steps": 20}
}

cases_parabolic = {
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


cases_flow = [
    {"label": "5 slm", "prefix": "plt_N2_5slm_600W_", "start": 64500, "stride": 500, "n": 20, "flow_val": 5.0, "power": 600},
    {"label": "10 slm", "prefix": "plt_N2_10slm_600W_", "start": 66500, "stride": 500, "n": 20, "flow_val": 10.0, "power": 600},
    {"label": "15 slm", "prefix": "plt_N2_15slm_600W_", "start": 68000, "stride": 500, "n": 20, "flow_val": 15.0, "power": 600},
    {"label": "20 slm", "prefix": "plt_N2_20slm_600W_", "start": 71000, "stride": 500, "n": 21, "flow_val": 20.0, "power": 600},
]


# ---------------------------------------------------------------- extraction
def find_plotfile(prefix, step):
    for name in (f"{prefix}{step:05d}", f"{prefix}{step}"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def slice_flows(path, z_target):
    ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
    lvl = ds.index.max_level; ref = int(2 ** lvl)
    dims = [ds.domain_dimensions[0] * ref, ds.domain_dimensions[1] * ref, 1]
    cg = ds.covering_grid(level=lvl, left_edge=ds.domain_left_edge, dims=dims)
    le, re = ds.domain_left_edge.v, ds.domain_right_edge.v
    dr, dz = (re[0] - le[0]) / dims[0], (re[1] - le[1]) / dims[1]
    r = le[0] + (np.arange(dims[0]) + 0.5) * dr
    z = le[1] + (np.arange(dims[1]) + 0.5) * dz
    iz = int(np.argmin(np.abs(z - z_target)))
    vz, T, yn = cg["y_velocity"][:, iz, 0].v, cg["temp"][:, iz, 0].v, cg["Y(N)"][:, iz, 0].v
    mask = (T >= T_MIN) & (r <= R_TUBE)
    dA = 2.0 * np.pi * r * dr * mask
    M_mix = 1.0 / (yn / M_N + (1.0 - yn) / M_N2)
    rho = P_ATM * M_mix / (R_GAS * T)
    n_dot_N = np.sum(rho * yn * vz * dA) / M_N
    return {"N_sccm": n_dot_N * 60.0 * V_MOL}


def extract_n_flow_sccm(case_dict, z_target):
    powers, mean = [], []
    for p in sorted(case_dict):
        c = case_dict[p]
        steps = [c["start_step"] - i * SNAP_DT for i in range(c["n_steps"])]
        flows = []
        for s in steps:
            path = find_plotfile(c["cfd"], s)
            if path is None:
                continue
            out = slice_flows(path, z_target)
            flows.append(out["N_sccm"])
        if not flows:
            continue
        powers.append(p); mean.append(np.mean(flows))
    return np.array(powers), np.array(mean)


def energy_cost(power_W, flow_sccm):
    flow_sccm = np.asarray(flow_sccm, dtype=float)
    mol_per_s = flow_sccm / V_MOL / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = np.where(mol_per_s > 0, power_W / mol_per_s / 1e6, np.nan)
    return cost


def z_profile_flows(path, z_min=0.08, z_max=0.27):
    ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
    lvl = ds.index.max_level; ref = int(2 ** lvl)
    dims = [ds.domain_dimensions[0] * ref, ds.domain_dimensions[1] * ref, 1]
    cg = ds.covering_grid(level=lvl, left_edge=ds.domain_left_edge, dims=dims)
    le, re = ds.domain_left_edge.v, ds.domain_right_edge.v
    dr, dz = (re[0] - le[0]) / dims[0], (re[1] - le[1]) / dims[1]
    r = le[0] + (np.arange(dims[0]) + 0.5) * dr
    z = le[1] + (np.arange(dims[1]) + 0.5) * dz
    mask_z = (z >= z_min) & (z <= z_max)
    z_vals = z[mask_z]
    z_indices = np.where(mask_z)[0]
    flows = []
    for iz in z_indices:
        vz, T, yn = cg["y_velocity"][:, iz, 0].v, cg["temp"][:, iz, 0].v, cg["Y(N)"][:, iz, 0].v
        mask = (T >= T_MIN) & (r <= R_TUBE)
        dA = 2.0 * np.pi * r * dr * mask
        M_mix = 1.0 / (yn / M_N + (1.0 - yn) / M_N2)
        rho = P_ATM * M_mix / (R_GAS * T)
        n_dot_N = np.sum(rho * yn * vz * dA) / M_N
        flows.append(n_dot_N * 60.0 * V_MOL)
    return z_vals, np.array(flows)


def extract_z_profile(case_info, z_min=0.08, z_max=0.27):
    steps = [case_info["start_step"] - i * SNAP_DT for i in range(case_info["n_steps"])]
    profile_list = []
    z_vals = None
    for s in steps:
        path = find_plotfile(case_info["cfd"], s)
        if path is None:
            continue
        zv, flows = z_profile_flows(path, z_min, z_max)
        if z_vals is None:
            z_vals = zv
        profile_list.append(flows)
    if not profile_list:
        return None, None
    return z_vals, np.mean(profile_list, axis=0)


def window(cost):
    """Hide values outside the visible energy-cost window (out-of-plasma spikes)."""
    return np.where((cost >= COST_MIN) & (cost <= COST_MAX), cost, np.nan)


# ================================================================ extraction run
pw_h, flow_h = extract_n_flow_sccm(cases_harps, Z_TARGET)
pw_p, flow_p = extract_n_flow_sccm(cases_parabolic, Z_TARGET)
pw_c, flow_c = extract_n_flow_sccm(cases_cylinder, Z_TARGET)
cost_h = energy_cost(pw_h, flow_h)
cost_p = energy_cost(pw_p, flow_p)
cost_c = energy_cost(pw_c, flow_c)

# ======================================================== Figure 1: power sweep
fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
ax2 = ax.twinx()
ax.plot(pw_h, flow_h, "o-", color=BLUE)
ax.plot(pw_p, flow_p, "s--", color=BLUE, mfc="white")
ax.plot(pw_c, flow_c, "^:", color=BLUE, mfc="white")
ax2.plot(pw_h, cost_h, "o-", color=RED)
ax2.plot(pw_p, cost_p, "s--", color=RED, mfc="white")
ax2.plot(pw_c, cost_c, "^:", color=RED, mfc="white")
ax.set_yscale("log"); ax2.set_yscale("log")
ax.set_xlabel("Input Power (W)")
ax.set_ylabel(rf"N atom flow rate [sccm]", color=BLUE)
ax2.set_ylabel(r"Energy cost [MJ mol$^{-1}$ N]", color=RED)
ax.tick_params(axis="y", colors=BLUE); ax2.tick_params(axis="y", colors=RED)
ax.spines["left"].set_color(BLUE); ax2.spines["right"].set_color(RED)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":"); ax.margins(x=0.04)
handles = [
    Line2D([], [], color="black", marker="o", ls="-", label="EM solver"),
    Line2D([], [], color="black", marker="s", ls="--", mfc="white", label="Parabolic"),
    Line2D([], [], color="black", marker="^", ls=":", mfc="white", label="Cylinder"),
]
ax.legend(
    handles=handles, loc="center right", frameon=True,
    framealpha=0.9, fontsize=12, title="Power Profiles"
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("analysis_outputs/figure_n_atoms_power.png", dpi=300, bbox_inches="tight")
fig.savefig("analysis_outputs/figure_n_atoms_power.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================== Figure 2: inlet flow rate
flow_vals, flow_n_mean, flow_costs = [], [], []
for item in cases_flow:
    steps = [item["start"] - i * item["stride"] for i in range(item["n"])]
    flows = []
    for s in steps:
        path = find_plotfile(item["prefix"], s)
        if path is not None:
            out = slice_flows(path, Z_TARGET)
            flows.append(out["N_sccm"])
    if flows:
        mean_val = np.mean(flows)
        flow_vals.append(item["flow_val"])
        flow_n_mean.append(mean_val)
        flow_costs.append(float(energy_cost(item["power"], mean_val)))
flow_vals = np.array(flow_vals); flow_n_mean = np.array(flow_n_mean); flow_costs = np.array(flow_costs)

fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
ax2 = ax.twinx()
ax.plot(flow_vals, flow_n_mean, "o-", color=BLUE)
ax2.plot(flow_vals, flow_costs, "o-", color=RED)
ax.set_xlabel("Inlet Flow Rate (slm)")
ax.set_ylabel(rf"N atom flow rate [sccm]", color=BLUE)
ax2.set_ylabel(r"Energy cost [MJ mol$^{-1}$ N]", color=RED)
ax.tick_params(axis="y", colors=BLUE); ax2.tick_params(axis="y", colors=RED)
ax.spines["left"].set_color(BLUE); ax2.spines["right"].set_color(RED)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":"); ax.margins(x=0.08)
fig.tight_layout()
fig.savefig("analysis_outputs/figure_n_atoms_flow.png", dpi=300, bbox_inches="tight")
fig.savefig("analysis_outputs/figure_n_atoms_flow.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 3: z-profile, 600 / 1000 / 1500 W with extended lighter grey shading
z_600, flow_z_600 = extract_z_profile(cases_harps[600], 0.095, 0.27)
z_1000, flow_z_1000 = extract_z_profile(cases_harps[1000], 0.095, 0.27)
z_1500, flow_z_1500 = extract_z_profile(cases_harps[1500], 0.095, 0.27)
if z_600 is None or z_1000 is None or z_1500 is None:
    raise FileNotFoundError("z-profile plotfiles not found for the 600 W, 1000 W, and/or 1500 W EM-solver case.")

cost_z_600 = window(energy_cost(600, flow_z_600))
cost_z_1000 = window(energy_cost(1000, flow_z_1000))
cost_z_1500 = window(energy_cost(1500, flow_z_1500))

fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
ax2 = ax.twinx()

# Shaded regions: 2 cm extra on each side (10.44 to 12.44 cm and 16.76 to 18.76 cm) in lighter grey, core in original grey
ax.axvspan(10.44, 12.44, color="lightgray", alpha=0.2, zorder=0)
ax.axvspan(16.76, 18.76, color="lightgray", alpha=0.2, zorder=0)
ax.axvspan(12.44, 16.76, color="lightgray", alpha=0.5, zorder=0)

ax.plot(z_600*100, flow_z_600, "-", color=BLUE, zorder=3)
ax.plot(z_1000*100, flow_z_1000, "--", color=BLUE, zorder=3)
ax.plot(z_1500*100, flow_z_1500, "-.", color=BLUE, zorder=3)
ax2.plot(z_600*100, cost_z_600, "-", color=RED, zorder=3)
ax2.plot(z_1000*100, cost_z_1000, "--", color=RED, zorder=3)
ax2.plot(z_1500*100, cost_z_1500, "-.", color=RED, zorder=3)

ax2.set_yscale("log")
ax2.set_ylim(COST_MIN, COST_MAX)
ax2.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_locator(NullLocator())

ax.set_ylim(bottom=0)
ax.set_xlabel("Axial Position $z$ (cm)")
ax.set_ylabel(r"N atom flow rate [sccm]", color=BLUE)
ax2.set_ylabel(r"Energy cost [MJ mol$^{-1}$ N]", color=RED)
ax.tick_params(axis="y", colors=BLUE); ax2.tick_params(axis="y", colors=RED)
ax.spines["left"].set_color(BLUE); ax2.spines["right"].set_color(RED)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":"); ax.margins(x=0.04)

handles = [
    Line2D([], [], color="black", ls="-", label="600 W"),
    Line2D([], [], color="black", ls="--", label="1000 W"),
    Line2D([], [], color="black", ls="-.", label="1500 W"),
]
ax.legend(
    handles=handles, loc="center right", frameon=True,
    framealpha=0.9, fontsize=12, title="Input Power"
)
fig.tight_layout()
fig.savefig("analysis_outputs/figure_n_atoms_z_profile.png", dpi=300, bbox_inches="tight")
fig.savefig("analysis_outputs/figure_n_atoms_z_profile.pdf", bbox_inches="tight")
plt.close(fig)


z_5, flow_z_5 = extract_z_profile({"start_step": cases_flow[0]["start"], "n_steps": cases_flow[0]["n"], "cfd": cases_flow[0]["prefix"]}, 0.095, 0.27)
z_10, flow_z_10 = extract_z_profile({"start_step": cases_flow[1]["start"], "n_steps": cases_flow[1]["n"], "cfd": cases_flow[1]["prefix"]}, 0.095, 0.27)
z_20, flow_z_20 = extract_z_profile({"start_step": cases_flow[3]["start"], "n_steps": cases_flow[3]["n"], "cfd": cases_flow[3]["prefix"]}, 0.095, 0.27)
if z_5 is None or z_10 is None or z_20 is None:
    raise FileNotFoundError("z-profile plotfiles not found for the flow cases.")

cost_z_5 = window(energy_cost(cases_flow[0]["power"], flow_z_5))
cost_z_10 = window(energy_cost(cases_flow[1]["power"], flow_z_10))
cost_z_20 = window(energy_cost(cases_flow[3]["power"], flow_z_20))

fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
ax2 = ax.twinx()
ax.axvspan(10.44, 12.44, color="lightgray", alpha=0.2, zorder=0)
ax.axvspan(16.76, 18.76, color="lightgray", alpha=0.2, zorder=0)
ax.axvspan(12.44, 16.76, color="lightgray", alpha=0.5, zorder=0)
ax.plot(z_5*100, flow_z_5, "-", color=BLUE, zorder=3)
ax.plot(z_10*100, flow_z_10, "--", color=BLUE, zorder=3)
ax.plot(z_20*100, flow_z_20, "-.", color=BLUE, zorder=3)
ax2.plot(z_5*100, cost_z_5, "-", color=RED, zorder=3)
ax2.plot(z_10*100, cost_z_10, "--", color=RED, zorder=3)
ax2.plot(z_20*100, cost_z_20, "-.", color=RED, zorder=3)
ax2.set_yscale("log")
ax2.set_ylim(COST_MIN, COST_MAX)
ax2.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(bottom=0)
ax.set_xlabel("Axial Position $z$ (cm)")
ax.set_ylabel(r"N atom flow rate [sccm]", color=BLUE)
ax2.set_ylabel(r"Energy cost [MJ mol$^{-1}$ N]", color=RED)
ax.tick_params(axis="y", colors=BLUE); ax2.tick_params(axis="y", colors=RED)
ax.spines["left"].set_color(BLUE); ax2.spines["right"].set_color(RED)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":"); ax.margins(x=0.04)
handles = [
    Line2D([], [], color="black", ls="-", label="5 slm"),
    Line2D([], [], color="black", ls="--", label="10 slm"),
    Line2D([], [], color="black", ls="-.", label="20 slm"),
]
ax.legend(
    handles=handles, loc="center right", frameon=True,
    framealpha=0.9, fontsize=12, title="Inlet Flow Rate"
)
fig.tight_layout()
fig.savefig("analysis_outputs/figure_n_atoms_z_profile_flow.png", dpi=300, bbox_inches="tight")
fig.savefig("analysis_outputs/figure_n_atoms_z_profile_flow.pdf", bbox_inches="tight")
plt.close(fig)