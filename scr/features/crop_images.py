import rasterio
from rasterio.windows import Window
import os
import numpy as np

# add texture bands to the rasters before cropping
def add_texture_bands(folder_image_path, folder_texture_path, output_folder):
    """
    Add texture bands to the rasters in the specified folder and save the new rasters to the specified output folder, with the texture bands already in a given folder.
    """
    # Get a list of all files in the folder
    files = os.listdir(folder_image_path)
    # Filter out non-image files
    image_files = [f for f in files if f.endswith(('.tif', '.tiff'))]
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Loop through each image file
    for file_name in image_files:
        # Open the image file
        image_path = os.path.join(folder_image_path, file_name)
        texture_path_energy = os.path.join(folder_texture_path, file_name.replace('.tif', '_energy.npy'))
        texture_path_homogeneity = os.path.join(folder_texture_path, file_name.replace('.tif', '_homogeneity.npy'))

        with rasterio.open(image_path) as src:
            # Read the TIFF image
            nir, red, green, blue = src.read()
            # Get the metadata for the image
            meta = src.meta.copy()
            # Read the texture bands
            texture_energy = np.load(texture_path_energy)
            texture_homogeneity = np.load(texture_path_homogeneity)
            # Stack the texture bands with the image bands
            nir = nir[1:nir.shape[0]-1, 1:nir.shape[1]-1]
            red = red[1:red.shape[0]-1, 1:red.shape[1]-1]
            green = green[1:green.shape[0]-1, 1:green.shape[1]-1]
            blue = blue[1:blue.shape[0]-1, 1:blue.shape[1]-1]
            print(nir.shape, red.shape, green.shape, blue.shape, texture_energy.shape, texture_homogeneity.shape)
            # reduce the bands to the same size

            img_with_texture = np.stack([nir, red, green, blue, texture_energy, texture_homogeneity])
            # Update metadata with new count
            meta['count'] = img_with_texture.shape[0]
            # Construct the output file path
            output_file_path = os.path.join(output_folder, f"with_texture_{file_name}")
            # Write the image with texture bands to the output folder
            with rasterio.open(output_file_path, 'w', **meta) as dst:
                dst.write(img_with_texture)
        



def crop_images(folder_path, coordinates, output_folder):
    """
    Crop all images in the specified folder using the given coordinates,
    and save the cropped images to the specified output folder.

    Args:
    - folder_path: The path to the folder containing images.
    - coordinates: A tuple containing the left, upper, right, and lower coordinates of the cropping box.
    - output_folder: The path to the folder where cropped images will be saved.
    """
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out non-image files
    image_files = [f for f in files if f.endswith(('.tif', '.tiff'))]
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through each image file
    for file_name in image_files:
        # Open the image file
        image_path = os.path.join(folder_path, file_name)
        with rasterio.open(image_path) as src:
            # Read the TIFF image
            img = src.read()
            
            # Get the metadata for the image
            meta = src.meta.copy()
            
            # Calculate the window to crop
            left, upper, right, lower = coordinates
            window = Window.from_slices((upper, lower), (left, right))
            
            # Crop the image using the specified window
            cropped_img = src.read(window=window)
            
            # Update metadata with new height and width
            meta['height'], meta['width'] = cropped_img.shape[-2], cropped_img.shape[-1]
            
            # Construct the output file path
            output_file_path = os.path.join(output_folder, f"cropped_{file_name}")
            
            # Write the cropped image to the output folder
            with rasterio.open(output_file_path, 'w', **meta) as dst:
                dst.write(cropped_img)

# Example usage
folder_path = "data/raw"
texture_input_folder = "data/textures"
texture_output_folder = "data/with_texture"
# add_texture_bands(folder_path, texture_input_folder, texture_output_folder)

output_folder = "data/cropped"
coordinates = (430, 380, 480, 400)  # Example coordinates (left, upper, right, lower)
crop_images(texture_output_folder, coordinates, output_folder)
