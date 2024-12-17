import os
import re
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def extract_coords_and_ent(img_dir):
    filename_pattern = re.compile(r"([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+).*_ENT=(-?\d*\.?\d+)")
    coords_and_ent = []

    for f in os.listdir(img_dir):
        if f.endswith(".png"):
            m = filename_pattern.search(f)
            if m:
                x_str, y_str, ent_str = m.groups()
                x = float(x_str)
                y = -float(y_str)  # Invert y-values
                ent = float(ent_str)
                coords_and_ent.append((x, y, ent))
    return coords_and_ent

def create_ent_map(img_dir, output_name, resolution=1024, cmap_name='Greys'):
    # Extract coordinates and ENT values
    data = extract_coords_and_ent(img_dir)
    if not data:
        raise ValueError("No coordinates and ENT values found.")
    data = np.array(data)
    xs = data[:,0]
    ys = data[:,1]
    ents = data[:,2]

    # Determine bounding box
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    # Create a regular grid over the bounding box
    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x, max_x, resolution),
        np.linspace(min_y, max_y, resolution)
    )

    # Perform linear interpolation on ENT values
    grid_ent = griddata((xs, ys), ents, (grid_x, grid_y), method='linear')

    # Keep track of where we have NaN values
    nan_mask = np.isnan(grid_ent)

    # For normalization purposes, replace NaNs temporarily with something benign (like the min ENT).
    # The alpha channel will be used to represent transparency for these NaN points.
    ent_min = np.nanmin(grid_ent[~nan_mask]) if not np.all(nan_mask) else 0.0
    ent_max = np.nanmax(grid_ent[~nan_mask]) if not np.all(nan_mask) else 1.0

    # Replace NaNs with ent_min for normalization (their alpha will be set to 0 later)
    grid_ent_nonan = grid_ent.copy()
    grid_ent_nonan[nan_mask] = ent_min

    # Normalize ENT values to [0,1]
    normalized_ent = (grid_ent_nonan - ent_min) / (ent_max - ent_min + 1e-9)

    # Apply a colormap to the ENT values (returns RGBA)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colored_img = cmap(normalized_ent) # shape: (resolution, resolution, 4)

    # Convert to uint8
    colored_img = (colored_img * 255).astype(np.uint8)

    # Set alpha=0 where nan_mask is True (making those pixels transparent)
    colored_img[nan_mask, 3] = 0

    # Create an RGBA image from the array
    output_img = Image.fromarray(colored_img, 'RGBA')
    output_img.save(output_name)
    print(f"Saved interpolated ENT image to {output_name}")

# Example usage:
create_ent_map("./hpsearch/renders/", "assembled_entropy.png", resolution=2560)
