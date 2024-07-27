from PIL import Image
import numpy as np
from stl import mesh
import tkinter as tk
from tkinter import filedialog, messagebox

def create_lithophane_with_layers(
    image_path, 
    output_path, 
    max_thickness=3.04, 
    min_thickness=0.8, 
    num_color_layers=4, 
    color_layer_thickness=0.44, 
    base_thickness=0.2, 
    size=(86, 86),
    lithophane_resolution=0.08
):
    try:
        # Open and process the image
        image = Image.open(image_path).convert('L')
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

        # Create vertices and faces for lithophane
        vertices = np.zeros((rows * cols * 2, 3))
        for i in range(rows):
            for j in range(cols):
                vertices[i * cols + j] = [
                    j * lithophane_resolution, 
                    i * lithophane_resolution, 
                    thickness[i, j] + base_thickness
                ]
                vertices[rows * cols + i * cols + j] = [
                    j * lithophane_resolution, 
                    i * lithophane_resolution, 
                    0
                ]

        faces = []
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

                faces.append([top_v1, top_v3, top_v4])
                faces.append([top_v1, top_v4, top_v2])
                faces.append([bottom_v1, bottom_v4, bottom_v3])
                faces.append([bottom_v1, bottom_v2, bottom_v4])
                faces.append([top_v1, bottom_v1, bottom_v2])
                faces.append([top_v1, bottom_v2, top_v2])
                faces.append([top_v2, bottom_v2, bottom_v4])
                faces.append([top_v2, bottom_v4, top_v4])
                faces.append([top_v4, bottom_v4, bottom_v3])
                faces.append([top_v4, bottom_v3, top_v3])
                faces.append([top_v3, bottom_v3, bottom_v1])
                faces.append([top_v3, bottom_v1, top_v1])

        faces = np.array(faces)
        lithophane_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                lithophane_mesh.vectors[i][j] = vertices[face[j], :]

        lithophane_output_path = f"{output_path}_lithophane.stl"
        lithophane_mesh.save(lithophane_output_path)
        print(f"Lithophane saved to {lithophane_output_path}")

        # Create color layers
        create_color_layers(
            image, 
            output_path, 
            num_color_layers, 
            color_layer_thickness, 
            base_thickness, 
            lithophane_resolution, 
            size
        )

    except Exception as e:
        print(f"An error occurred: {e}")

def create_color_layers(
    image, 
    output_path, 
    num_layers=4, 
    color_layer_thickness=0.44, 
    base_thickness=0.2, 
    lithophane_resolution=0.08, 
    size=(86, 86)
):
    try:
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

            vertices = np.zeros((rows * cols * 2, 3))
            faces = []
            
            for i in range(rows):
                for j in range(cols):
                    if layer_mask[i, j]:
                        vertices[i * cols + j] = [
                            j * lithophane_resolution, 
                            i * lithophane_resolution, 
                            base_thickness + layer * color_layer_thickness
                        ]
                        vertices[rows * cols + i * cols + j] = [
                            j * lithophane_resolution, 
                            i * lithophane_resolution, 
                            base_thickness + (layer + 1) * color_layer_thickness
                        ]

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

                        faces.append([top_v1, top_v3, top_v4])
                        faces.append([top_v1, top_v4, top_v2])
                        faces.append([bottom_v1, bottom_v4, bottom_v3])
                        faces.append([bottom_v1, bottom_v2, bottom_v4])
                        faces.append([top_v1, bottom_v1, bottom_v2])
                        faces.append([top_v1, bottom_v2, top_v2])
                        faces.append([top_v2, bottom_v2, bottom_v4])
                        faces.append([top_v2, bottom_v4, top_v4])
                        faces.append([top_v4, bottom_v4, bottom_v3])
                        faces.append([top_v4, bottom_v3, top_v3])
                        faces.append([top_v3, bottom_v3, bottom_v1])
                        faces.append([top_v3, bottom_v1, top_v1])

            faces = np.array(faces)
            color_layer_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, face in enumerate(faces):
                for j in range(3):
                    color_layer_mesh.vectors[i][j] = vertices[face[j], :]

            color_name = color_map.get(layer, f'color_{layer}')
            color_layer_output_path = f"{output_path}_{color_name}.stl"
            color_layer_mesh.save(color_layer_output_path)
            print(f"{color_name.capitalize()} layer saved to {color_layer_output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

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

def main():
    image_paths = select_files()
    if not image_paths:
        messagebox.showerror("Error", "No files selected.")
        return

    output_path = select_output_path()
    if not output_path:
        messagebox.showerror("Error", "No output path selected.")
        return

    for image_path in image_paths:
        create_lithophane_with_layers(
            image_path,
            output_path.replace(".stl", f"_{image_path.split('/')[-1].split('.')[0]}"),
            max_thickness=3.04, 
            min_thickness=0.8,
            num_color_layers=4, 
            color_layer_thickness=0.44,
            base_thickness=0.2, 
            size=(86, 86)
        )

if __name__ == "__main__":
    main()
