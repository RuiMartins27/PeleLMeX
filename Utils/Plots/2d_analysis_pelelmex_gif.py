import yt
import imageio.v2 as imageio
import os

# ==========================================
# 1. Configuration
# ==========================================
base_path = "../../Exec/Production/2DSwirlCase/plt_N2_10slm_600W_"
#base_path = "../../Exec/Production/2DSwirlCase/plt_harps_"

full_sequence = list(range(0, 40000, 500))

# Define the quantities to plot and their specific settings
plots_config = {
    "temp": {"log_scale": False, "fix_scale": False},
    #"y_velocity": {"log_scale": False, "fix_scale": True},
    #"AngMom": {"log_scale": True, "fix_scale": False},
    "extsource_rhoh": {"log_scale": False, "fix_scale": False},
    #"Y(N)": {"log_scale": False, "fix_scale": False}
}

if "plt_harps" in base_path:
    plots_config = {
        "electron_temp": {"log_scale": False, "fix_scale": False},
        "E_over_N": {"log_scale": False, "fix_scale": False},
        "electron_dens": {"log_scale": False, "fix_scale": False}
    }


# Define z-limits for known quantities (only applied if fix_scale is True)
z_limits = {
    "temp": (275, 6000),
    "z_velocity": (-15, 15),
    "y_velocity": (-5, 5),
    "x_velocity": (-1, 1),
    "AngMom": (0.0, 0.01),
    "avg_pressure": (-0.15, 0.15),
    "extsource_rhoh": (0, 1.1e8),
    "Y(O2)": (0, 1.0),
    "Y(N)": (0, 0.10)
}

frames = {quantity: [] for quantity in plots_config.keys()}


print(f"Processing {len(full_sequence)} potential frames...")

for num in full_sequence:
    padded_num = f"{num:05d}"
    file_path = f"{base_path}{padded_num}"
    
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}: Folder not found.")
        continue

    print(f"\n--- Loading frame: {padded_num} ---")
    ds = yt.load(file_path, units_override={'length_unit': (1.0, 'm')})
    
    for quantity, settings in plots_config.items():
        log_scale = settings["log_scale"]
        fix_scale = settings["fix_scale"]

        if "Y(" in quantity or "dens" in quantity:
            cmap = "viridis"
        elif "velocity" in quantity:
            cmap = "RdBu_r"
        elif "AngMom" in quantity or "E_over_N" in quantity: 
            cmap = "plasma"
        elif "extsource" in quantity:
            cmap = "inferno"
        else:
            cmap = "magma"

        slc = yt.SlicePlot(ds, 'theta', quantity, origin='native')
        slc.swap_axes() 
        
        # Apply Log Scale (with symlog fallback for fix_scale=False)
        if log_scale and not fix_scale:
            slc.set_log(quantity, True, linthresh=1e-6)
        else:
            slc.set_log(quantity, log_scale)

        if fix_scale and quantity in z_limits:
            z_min, z_max = z_limits[quantity]
            if log_scale and z_min <= 0:
                z_min = 1e-6 
                
            slc.set_zlim(quantity, z_min, z_max)

        slc.set_cmap(quantity, cmap)
        slc.set_font({'size': 12, 'family': 'serif'})
        
        temp_prefix = f"temp_{quantity}_frame_{padded_num}"
        saved_paths = slc.save(temp_prefix)
        actual_frame_path = saved_paths[0]
        
        frames[quantity].append(imageio.imread(actual_frame_path))
        os.remove(actual_frame_path)


print("\n=== Generating GIFs ===")
for quantity, frame_list in frames.items():
    if frame_list:
        gif_name = f'2d_{quantity}_evolution.gif'
        imageio.mimsave(gif_name, frame_list, duration=100, loop=0)
        print(f"Success: {gif_name}")
    else:
        print(f"Warning: No frames were processed for {quantity}.")