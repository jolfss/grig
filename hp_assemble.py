import os
from PIL import Image
import re

def assemble(img_dir, output_name, tile_size=350, num_tiles=100):

    # Pattern to extract x, y coordinates from filenames.
    filename_pattern = re.compile(r"([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_")

    # Collect all coordinates and corresponding filenames
    coords = []
    png_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    for f in png_files:
        m = filename_pattern.search(f)
        if not m:
            continue
        x_str, y_str = m.groups()
        x = float(x_str)
        y = float(y_str)
        coords.append((x, y, f))

    if not coords:
        raise ValueError("No valid coordinates found in filenames.")

    # Determine the bounding box of all coordinates
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Decide on a scale factor
    desired_width = tile_size*num_tiles/1.90  # adjust as needed
    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range == 0 or y_range == 0:
        raise ValueError("All coordinates are the same or no range in one dimension.")

    scale = desired_width / x_range

    padding = tile_size
    img_width = int(x_range * scale) + padding
    img_height = int(y_range * scale) + padding

    offset_x = -int(min_x * scale) + padding // 2

    offset_y = padding // 2

    # Create a canvas with RGBA so transparency is supported
    canvas = Image.new("RGBA", (img_width, img_height), (0,0,0,0))

    for (x_coord, y_coord, filename) in coords:
        # Convert coordinates to pixel space
        px = int((x_coord - min_x)*scale) + padding // 2
        # Invert y: top corresponds to max_y, bottom to min_y
        py = int((max_y - y_coord) * scale) + offset_y

        # Center the tile at (px, py)
        px -= tile_size // 2
        py -= tile_size // 2

        img_path = os.path.join(img_dir, filename)
        sub_img = Image.open(img_path).convert("RGBA")

        # Replace near-black pixels (R,G,B < 10) with transparent
        datas = sub_img.getdata()
        new_data = []
        for item in datas:
            # Check if all channels are less than 10
            if (item[0] < 10 and item[1] < 10 and item[2] < 10):
                new_data.append((0,0,0,0))
            else:
                new_data.append(item)
        sub_img.putdata(new_data)

        # Paste onto the canvas with transparency
        canvas.paste(sub_img, (px, py), sub_img)

    canvas.save(output_name)

assemble("./hpsearch/renders/","assembled.png")