# file_path: lithophane_generator.py

from PIL import Image, ImageTk
import numpy as np
import trimesh
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_lithophane_with_layers(
    image_path, 
    output_path, 
    max_thickness=3.5, 
    min_thickness=0.8, 
    num_color_layers=4, 
    color_layer_thickness=0.44, 
    base_thickness=0.2, 
    size=(86, 86),
    lithophane_resolution=0.08
):
    try:
        image = Image.open(image_path).convert('L')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {e}")
        return

    image.thumbnail(size, Image.LANCZOS)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    pixel_values = np.array(image)
    
    thickness = np.interp(
        pixel_values, 
        (pixel_values.min(), pixel_values.max()), 
        (max_thickness, min_thickness)
    )

    rows, cols = thickness.shape
    width_mm = cols * lithophane_resolution
    height_mm = rows * lithophane_resolution

    vertices = []
    faces = []

    for i in range(rows):
        for j in range(cols):
            vertices.append([j * lithophane_resolution, i * lithophane_resolution, thickness[i, j] + base_thickness])
            vertices.append([j * lithophane_resolution, i * lithophane_resolution, 0])

    for i in range(rows - 1):
        for j in range(cols - 1):
            top_v1 = i * cols + j
            top_v2 = top_v1 + 1
            top_v3 = top_v1 + cols
            top_v4 = top_v3 + 1

            bottom_v1 = rows * cols + i * cols + j
            bottom_v2 = bottom_v1 + 1
            bottom_v3 = bottom_v1 + cols
            bottom_v4 = bottom_v3 + 1

            faces.extend([
                [top_v1, top_v3, top_v4],
                [top_v1, top_v4, top_v2],
                [bottom_v1, bottom_v4, bottom_v3],
                [bottom_v1, bottom_v2, bottom_v4],
                [top_v1, bottom_v1, bottom_v2],
                [top_v1, bottom_v2, top_v2],
                [top_v2, bottom_v2, bottom_v4],
                [top_v2, bottom_v4, top_v4],
                [top_v4, bottom_v4, bottom_v3],
                [top_v4, bottom_v3, top_v3],
                [top_v3, bottom_v3, bottom_v1],
                [top_v3, bottom_v1, top_v1]
            ])

    vertices = np.array(vertices)
    faces = np.array(faces)
    lithophane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    lithophane_output_path = f"{output_path}_lithophane.stl"
    try:
        lithophane_mesh.export(lithophane_output_path)
        print(f"Lithophane saved to {lithophane_output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save lithophane: {e}")

    create_color_layers(
        image, 
        output_path, 
        num_color_layers, 
        color_layer_thickness, 
        base_thickness, 
        lithophane_resolution, 
        size
    )

    create_base_layer(output_path, width_mm, height_mm, base_thickness)

def create_color_layers(
    image, 
    output_path, 
    num_layers=4, 
    color_layer_thickness=0.44, 
    base_thickness=0.2, 
    lithophane_resolution=0.08, 
    size=(86, 86)
):
    pixel_values = np.array(image)
    min_pixel, max_pixel = pixel_values.min(), pixel_values.max()

    intervals = np.linspace(min_pixel, max_pixel, num_layers + 1)

    rows, cols = pixel_values.shape

    color_map = {
        0: 'cyan',
        1: 'yellow',
        2: 'magenta',
        3: 'white'
    }

    for layer in range(num_layers):
        lower_bound, upper_bound = intervals[layer], intervals[layer + 1]
        layer_mask = np.logical_and(pixel_values >= lower_bound, pixel_values < upper_bound)
        layer_thickness = np.where(layer_mask, color_layer_thickness, 0)

        if np.max(layer_thickness) == 0:
            print(f"Layer {layer + 1} is empty. Skipping...")
            continue

        vertices = []
        faces = []
        
        for i in range(rows):
            for j in range(cols):
                if layer_mask[i, j]:
                    vertices.append([j * lithophane_resolution, i * lithophane_resolution, base_thickness + layer * color_layer_thickness])
                    vertices.append([j * lithophane_resolution, i * lithophane_resolution, base_thickness + (layer + 1) * color_layer_thickness])

        for i in range(rows - 1):
            for j in range(cols - 1):
                if (layer_mask[i, j] and layer_mask[i + 1, j] and
                    layer_mask[i, j + 1] and layer_mask[i + 1, j + 1]):

                    top_v1 = i * cols + j
                    top_v2 = top_v1 + 1
                    top_v3 = top_v1 + cols
                    top_v4 = top_v3 + 1

                    bottom_v1 = rows * cols + i * cols + j
                    bottom_v2 = bottom_v1 + 1
                    bottom_v3 = bottom_v1 + cols
                    bottom_v4 = bottom_v3 + 1

                    faces.extend([
                        [top_v1, top_v3, top_v4],
                        [top_v1, top_v4, top_v2],
                        [bottom_v1, bottom_v4, bottom_v3],
                        [bottom_v1, bottom_v2, bottom_v4],
                        [top_v1, bottom_v1, bottom_v2],
                        [top_v1, bottom_v2, top_v2],
                        [top_v2, bottom_v2, bottom_v4],
                        [top_v2, bottom_v4, top_v4],
                        [top_v4, bottom_v4, bottom_v3],
                        [top_v4, bottom_v3, top_v3],
                        [top_v3, bottom_v3, bottom_v1],
                        [top_v3, bottom_v1, top_v1]
                    ])

        vertices = np.array(vertices)
        faces = np.array(faces)
        color_layer_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        color_name = color_map.get(layer, f'color_{layer}')
        color_layer_output_path = f"{output_path}_{color_name}.stl"
        try:
            color_layer_mesh.export(color_layer_output_path)
            print(f"{color_name.capitalize()} layer saved to {color_layer_output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save {color_name} layer: {e}")

def create_base_layer(output_path, width, height, thickness):
    vertices_base = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0],
        [0, 0, thickness],
        [width, 0, thickness],
        [width, height, thickness],
        [0, height, thickness]
    ])

    faces_base = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ])

    base_mesh = trimesh.Trimesh(vertices=vertices_base, faces=faces_base)
    base_output_path = f"{output_path}_base.stl"
    try:
        base_mesh.export(base_output_path)
        print(f"Base layer saved to {base_output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save base layer: {e}")

def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    return list(file_paths)

def select_output_path():
    root = tk.Tk()
    root.withdraw()
    output_path = filedialog.asksaveasfilename(
        defaultextension=".stl",
        filetypes=[("STL files", "*.stl")]
    )
    return output_path

def validate_input(value, input_type, min_value=None, max_value=None):
    try:
        value = input_type(value)
        if min_value is not None and value < min_value:
            raise ValueError
        if max_value is not None and value > max_value:
            raise ValueError
        return value
    except ValueError:
        messagebox.showerror("Invalid Input", f"Please enter a valid {input_type.__name__} between {min_value} and {max_value}.")
        return None

def get_user_input():
    root = tk.Tk()
    root.withdraw()

    while True:
        max_thickness = validate_input(
            simpledialog.askstring("Input", "Enter max thickness (default 3.5):", initialvalue="3.5"),
            float, 0.1, 10.0
        )
        if max_thickness is not None: break
    
    while True:
        min_thickness = validate_input(
            simpledialog.askstring("Input", "Enter min thickness (default 0.8):", initialvalue="0.8"),
            float, 0.1, 10.0
        )
        if min_thickness is not None: break
    
    while True:
        num_color_layers = validate_input(
            simpledialog.askstring("Input", "Enter number of color layers (default 4):", initialvalue="4"),
            int, 1, 10
        )
        if num_color_layers is not None: break
    
    while True:
        color_layer_thickness = validate_input(
            simpledialog.askstring("Input", "Enter color layer thickness (default 0.44):", initialvalue="0.44"),
            float, 0.1, 10.0
        )
        if color_layer_thickness is not None: break

    while True:
        base_thickness = validate_input(
            simpledialog.askstring("Input", "Enter base thickness (default 0.2):", initialvalue="0.2"),
            float, 0.1, 10.0
        )
        if base_thickness is not None: break
    
    while True:
        width = validate_input(
            simpledialog.askstring("Input", "Enter width (default 86):", initialvalue="86"),
            int, 10, 1000
        )
        if width is not None: break
    
    while True:
        height = validate_input(
            simpledialog.askstring("Input", "Enter height (default 86):", initialvalue="86"),
            int, 10, 1000
        )
        if height is not None: break
    
    while True:
        lithophane_resolution = validate_input(
            simpledialog.askstring("Input", "Enter lithophane resolution (default 0.08):", initialvalue="0.08"),
            float, 0.01, 1.0
        )
        if lithophane_resolution is not None: break
    
    params = {
        "max_thickness": max_thickness,
        "min_thickness": min_thickness,
        "num_color_layers": num_color_layers,
        "color_layer_thickness": color_layer_thickness,
        "base_thickness": base_thickness,
        "width": width,
        "height": height,
        "lithophane_resolution": lithophane_resolution
    }
    
    return params

def preview_lithophane(image_path, params):
    image = Image.open(image_path).convert('L')
    image.thumbnail((params['width'], params['height']), Image.LANCZOS)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    pixel_values = np.array(image)
    
    thickness = np.interp(
        pixel_values, 
        (pixel_values.min(), pixel_values.max()), 
        (params['max_thickness'], params['min_thickness'])
    )

    rows, cols = thickness.shape
    width_mm = cols * params['lithophane_resolution']
    height_mm = rows * params['lithophane_resolution']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Lithophane Preview')
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_zlabel('Thickness (mm)')
    
    for i in range(rows):
        for j in range(cols):
            ax.bar3d(
                j * params['lithophane_resolution'], 
                i * params['lithophane_resolution'], 
                0, 
                params['lithophane_resolution'], 
                params['lithophane_resolution'], 
                thickness[i, j] + params['base_thickness'], 
                shade=True
            )
    
    plt.show()

def interactive_preview(image_path):
    root = tk.Tk()
    root.title("Interactive Lithophane Preview")

    params = {
        "max_thickness": tk.DoubleVar(value=3.5),
        "min_thickness": tk.DoubleVar(value=0.8),
        "num_color_layers": tk.IntVar(value=4),
        "color_layer_thickness": tk.DoubleVar(value=0.44),
        "base_thickness": tk.DoubleVar(value=0.2),
        "width": tk.IntVar(value=86),
        "height": tk.IntVar(value=86),
        "lithophane_resolution": tk.DoubleVar(value=0.08)
    }

    def update_preview():
        nonlocal image_path, params
        params_dict = {key: var.get() for key, var in params.items()}
        preview_lithophane(image_path, params_dict)

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    for key, var in params.items():
        tk.Label(frame, text=key.replace('_', ' ').capitalize()).pack(anchor='w')
        if isinstance(var, tk.DoubleVar):
            ttk.Scale(frame, from_=0.1, to=10, variable=var, orient='horizontal').pack(fill='x', padx=5, pady=5)
        elif isinstance(var, tk.IntVar):
            ttk.Scale(frame, from_=1, to=1000, variable=var, orient='horizontal').pack(fill='x', padx=5, pady=5)

    preview_button = tk.Button(frame, text="Update Preview", command=update_preview)
    preview_button.pack(pady=10)

    root.mainloop()

def main():
    image_paths = select_files()
    if not image_paths:
        messagebox.showerror("Error", "No files selected.")
        return

    for image_path in image_paths:
        interactive_preview(image_path)
        output_path = select_output_path()
        if not output_path:
            messagebox.showerror("Error", "No output path selected.")
            return

        params = get_user_input()
        preview_lithophane(image_path, params)
        if messagebox.askyesno("Confirm", f"Do you want to save the lithophane for {image_path}?"):
            create_lithophane_with_layers(
                image_path,
                output_path.replace(".stl", f"_{image_path.split('/')[-1].split('.')[0]}"),
                max_thickness=params["max_thickness"], 
                min_thickness=params["min_thickness"],
                num_color_layers=params["num_color_layers"], 
                color_layer_thickness=params["color_layer_thickness"],
                base_thickness=params["base_thickness"], 
                size=(params["width"], params["height"]),
                lithophane_resolution=params["lithophane_resolution"]
            )

if __name__ == "__main__":
    main()
