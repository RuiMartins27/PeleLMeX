import os, yt, numpy as np, matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13, "lines.linewidth": 2.5, "lines.markersize": 6})
data_dir = "../../Exec/Production/2DSwirlCase"
cases_harps = {300: {"cfd": "plt_N2_10slm_300W_", "start_step": 56000, "n_steps": 17}, 450: {"cfd": "plt_N2_10slm_450W_", "start_step": 60000, "n_steps": 17}, 600: {"cfd": "plt_N2_10slm_600W_", "start_step": 66500, "n_steps": 20}, 800: {"cfd": "plt_N2_10slm_800W_", "start_step": 70000, "n_steps": 20}, 1000: {"cfd": "plt_N2_10slm_1000W_", "start_step": 76000, "n_steps": 23}, 1250: {"cfd": "plt_N2_10slm_1250W_", "start_step": 80000, "n_steps": 23}, 1500: {"cfd": "plt_N2_10slm_1500W_", "start_step": 85000, "n_steps": 23}}
cases_static = {300: {"cfd": "plt_N2_10slm_300W_static_", "start_step": 62000, "n_steps": 17}, 450: {"cfd": "plt_N2_10slm_450W_static_", "start_step": 74500, "n_steps": 20}, 600: {"cfd": "plt_N2_10slm_600W_static_", "start_step": 85000, "n_steps": 22}, 800: {"cfd": "plt_N2_10slm_800W_static_", "start_step": 106500, "n_steps": 24}, 1000: {"cfd": "plt_N2_10slm_1000W_static_", "start_step": 106500, "n_steps": 23}, 1250: {"cfd": "plt_N2_10slm_1250W_static_", "start_step": 106500, "n_steps": 24}, 1500: {"cfd": "plt_N2_10slm_1500W_static_", "start_step": 106500, "n_steps": 22}}

# Mass flow conversion to SCCM and energy cost verification
def extract_n_flow_sccm(case_dict, z_target=0.1981):
    powers, flows_sccm = [], []
    for p in sorted(case_dict.keys()):
        prefix, start_step, n_steps = case_dict[p]["cfd"], case_dict[p]["start_step"], case_dict[p]["n_steps"]
        flow_sum, count = 0.0, 0
        for s in [start_step - i * 500 for i in range(n_steps)]:
            path = os.path.join(data_dir, f"{prefix}{s:05d}") if not os.path.exists(os.path.join(data_dir, f"{prefix}{s}")) else os.path.join(data_dir, f"{prefix}{s}")
            if not os.path.exists(path): continue
            ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
            fine_dims = [ds.domain_dimensions[0] * int(2**ds.index.max_level), ds.domain_dimensions[1] * int(2**ds.index.max_level), 1]
            cg = ds.covering_grid(level=ds.index.max_level, left_edge=ds.domain_left_edge, dims=fine_dims)
            dz = (ds.domain_right_edge[1].v - ds.domain_left_edge[1].v) / fine_dims[1]
            z_idx = np.argmin(np.abs(np.linspace(ds.domain_left_edge[1].v + 0.5 * dz, ds.domain_right_edge[1].v - 0.5 * dz, fine_dims[1]) - z_target))
            rho, yn, vz = cg["density"].in_units("g/cm**3")[:, z_idx, 0].v, cg["Y(N)"][:, z_idx, 0].v, cg["y_velocity"].in_units("m/s")[:, z_idx, 0].v
            dr = (ds.domain_right_edge[0].v - ds.domain_left_edge[0].v) / fine_dims[0]
            r_1d = np.linspace(ds.domain_left_edge[0].v + 0.5 * dr, ds.domain_right_edge[0].v - 0.5 * dr, fine_dims[0])
            flow_sum += np.sum(rho * yn * vz * 2 * np.pi * r_1d * dr * 1e6) * (60.0 / 14.0067) * 22414.0
            count += 1
        if count > 0:
            powers.append(p)
            flows_sccm.append(flow_sum / count)
    return powers, flows_sccm

pw_harps, flow_harps = extract_n_flow_sccm(cases_harps)
pw_stat, flow_stat = extract_n_flow_sccm(cases_static)
energy_harps = np.array(pw_harps) / (np.array(flow_harps) / 22414.0 / 60.0) / 1e6
energy_stat = np.array(pw_stat) / (np.array(flow_stat) / 22414.0 / 60.0) / 1e6

fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
ax.plot(pw_harps, flow_harps, 'o-', color="#0055A5", label="N atoms (HARPS)")
ax.plot(pw_stat, flow_stat, 's--', color="#C8102E", label="N atoms (Static)")
ax.set_xlabel("Power (W)")
ax.set_ylabel(r"N Atom Flow at $z=0.1981$ (sccm)")
ax2 = ax.twinx()
ax2.plot(pw_harps, energy_harps, 'o-', color="#0055A5", alpha=0.7, label="Energy cost (HARPS)")
ax2.plot(pw_stat, energy_stat, 's--', color="#C8102E", alpha=0.7, label="Energy cost (Static)")
ax2.set_ylabel(r"Energy Cost (MJ mol$^{-1}$ N)")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
plt.tight_layout()
plt.savefig("figureX_N.png", dpi=300, bbox_inches="tight")