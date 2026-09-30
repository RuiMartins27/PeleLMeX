import os
import numpy as np
import matplotlib.pyplot as plt
import yt

# --- 1. Configuration & Dataset Discovery ---
base_dir = "../../Exec/Production/2DSwirlCase"
prefix = "plt_N2_600W_"
base_step = 0
offsets = [0, -250, -500, -750, -1000]

target_steps = [base_step + offset for offset in offsets]

axial_velocity_field = 'y_velocity' 
R_tube = 0.0135  # Tube radius boundary (meters)

# Pre-calculating standard condition conversion factor for Atmospheric Pressure:
conversion_factor = 1000 * 60 * 273.15 * 1

valid_datasets_flow = []
loaded_steps = []
common_z = None

print("Searching for 2D datasets...")

# --- 2. Load & Process Available 2D Datasets ---
for step in target_steps:
    # Handle zero-padded and non-zero-padded folder naming (e.g., plt01000 vs plt00000/plt0)
    possible_paths = [
        os.path.join(base_dir, f"{prefix}{step:05d}"),
        os.path.join(base_dir, f"{prefix}{step}")
    ]
    
    ds_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ds_path = path
            break

    if ds_path is None:
        print(f"  [Skipped] Dataset for step {step} not found.")
        continue

    print(f"  [Processing] Step {step} from {ds_path}...")
    try:
        ds = yt.load(ds_path, units_override={'length_unit': (1.0, 'm')})
    except Exception as e:
        print(f"  [Failed] Could not load dataset at {ds_path}: {e}")
        continue

    # Extract Uniform Finest-Level Grid
    max_level = ds.index.max_level
    ref_factor = int(2**max_level)
    fine_dims = [
        ds.domain_dimensions[0] * ref_factor, 
        ds.domain_dimensions[1] * ref_factor, 
        1
    ]

    cg = ds.covering_grid(level=max_level, left_edge=ds.domain_left_edge, dims=fine_dims)

    # Extract raw simulation data arrays (Shape: [radial_cells, axial_cells])
    vz_2d = cg[axial_velocity_field][:, :, 0].v   # Raw velocity values
    temp_2d = cg["temp"][:, :, 0].v               # K

    # Manually Reconstruct Clean Geometric Vectors (Meters)
    r_min = ds.domain_left_edge[0].v
    r_max = ds.domain_right_edge[0].v
    z_min = ds.domain_left_edge[1].v
    z_max = ds.domain_right_edge[1].v

    # Store z-coordinate array from the first successfully loaded dataset
    if common_z is None:
        common_z = np.linspace(z_min, z_max, fine_dims[1])

    # Calculate exact cell spacing along the radial axis
    dr = (r_max - r_min) / fine_dims[0]

    # Generate exact radial coordinates for cell centers
    r_1d = np.linspace(r_min + 0.5 * dr, r_max - 0.5 * dr, fine_dims[0])

    # Compute exact 3D axisymmetric area element for each ring
    dA_1d = 2 * np.pi * r_1d * dr

    # Expand geometry arrays to 2D for vector math
    r_2d = r_1d[:, np.newaxis]   # Shape: [radial_cells, 1]
    dA_2d = dA_1d[:, np.newaxis] # Shape: [radial_cells, 1]

    # Vectorized Integration
    local_flow_density = vz_2d * conversion_factor / temp_2d
    mask = (temp_2d >= 200.0) & (r_2d <= R_tube)

    flow_contribution = local_flow_density * dA_2d * mask
    flow_results = np.sum(flow_contribution, axis=0)  # Sum down radial axis

    valid_datasets_flow.append(flow_results)
    loaded_steps.append(step)

# --- 3. Ensemble Averaging ---
if not valid_datasets_flow:
    raise FileNotFoundError("No valid 2D datasets were found. Please check dataset file paths.")

# Convert to 2D matrix: shape (num_loaded_ds, num_z_cells)
flow_matrix = np.array(valid_datasets_flow)

# Compute mean and standard deviation across datasets for each axial point z
avg_flow = np.mean(flow_matrix, axis=0)
std_flow = np.std(flow_matrix, axis=0)

# --- 4. Visualization ---
plt.figure(figsize=(10, 6))

# Plot individual datasets as faint background lines
for step, flow in zip(loaded_steps, valid_datasets_flow):
    plt.plot(common_z, flow, linestyle='--', alpha=0.4, linewidth=1, label=f"Step {step}")

# Plot the ensemble average
plt.plot(common_z, avg_flow, linestyle='-', color='#1f77b4', linewidth=2.5, label='Ensemble Average')

# Add variance shading if multiple datasets are present
if len(loaded_steps) > 1:
    plt.fill_between(common_z, avg_flow - std_flow, avg_flow + std_flow, color='#1f77b4', alpha=0.15, label='±1 Std Dev')

plt.title(f'2D Axisymmetric Ensemble Average Standard Flow Rate (slm) vs. Z\n({len(loaded_steps)} Datasets Averaged)', fontsize=13, pad=15)
plt.xlabel('Z Position (m)', fontsize=12)
plt.ylabel('Flow Rate (slm)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)

plt.xlim(common_z[0], common_z[-1])
plt.legend(frameon=True, loc='best')
plt.tight_layout()
plt.savefig("2d_ensemble_flow.png", dpi=300)

# --- 5. Summary Statistics ---
print("\n" + "=" * 35)
print(f"Analysis Complete ({len(loaded_steps)} / {len(target_steps)} datasets loaded).")
print(f"Loaded steps: {loaded_steps}")
print("-" * 35)
print(f"Max Average Flow: {max(avg_flow):.3f} slm")
print(f"Min Average Flow: {min(avg_flow):.3f} slm")
print(f"Mean Flow across all positions: {np.mean(avg_flow):.3f} slm")
print("=" * 35)