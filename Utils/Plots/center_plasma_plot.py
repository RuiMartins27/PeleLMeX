import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

def parse_log(filename):
    times, x_vals, y_vals = [], [], []
    curr_t, curr_sdc = None, 0
    t_pat = re.compile(r"STEP\s*\[\d+\]\s*-\s*Time:\s*([0-9.eE+-]+)")
    sdc_pat = re.compile(r"SDC iter\s*\[(\d+)\]")
    c_pat = re.compile(r"Plasma center\s*=\s*\(\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)\s*\)")
    with open(filename, 'r') as f:
        for line in f:
            if "STEP" in line: curr_sdc = 0
            if (t := t_pat.search(line)): curr_t = float(t.group(1))
            if (s := sdc_pat.search(line)): curr_sdc = int(s.group(1))
            if (c := c_pat.search(line)) and curr_t is not None and curr_sdc != 2:
                times.append(curr_t); x_vals.append(float(c.group(1))); y_vals.append(float(c.group(2)))
    return np.array(times), np.array(x_vals), np.array(y_vals)

def plot_plasma_data(times, x, y):
    times, x, y = times[5:], x[5:] * 1000.0, y[5:] * 1000.0
    r, theta = np.sqrt(x**2 + y**2), np.degrees(np.arctan2(y, x))
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    sc = ax1.scatter(x, y, c=times, cmap='viridis', s=60, edgecolor='k', linewidth=0.4)
    cb = fig1.colorbar(sc)
    cb.ax.tick_params(labelsize=14)
    cb.set_label('Time (s)', fontsize=16)
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)
    ax1.set_xlabel('X Position (mm)'); ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Plasma Center Trajectory in XY Plane', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig1.tight_layout(); fig1.savefig("analysis_outputs/plasma_center_xy.png", dpi=300)
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    ax2.plot(times, r, color='tab:blue', linewidth=2.5, label='Radius')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Radius (mm)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax3 = ax2.twinx()
    ax3.plot(times, theta, color='tab:orange', linewidth=2.5, label='Angle')
    ax3.set_ylabel('Angle (degrees)', color='tab:orange')
    ax3.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_title('Plasma Center Radius and Angle vs Time', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.6)
    fig2.tight_layout(); fig2.savefig("analysis_outputs/plasma_center_time_series.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    base_dir = "../../Exec/Production/3D_Microwave"
    t, x, y = parse_log(base_dir + "/N2_3D_10slm_600W_-1490.out")
    if len(t) > 5: plot_plasma_data(t, x, y)
    else: print("Insufficient plasma center data found.")