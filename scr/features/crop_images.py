import rasterio
from rasterio.windows import Window
import os

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
output_folder = "data/cropped"
coordinates = (430, 380, 480, 400)  # Example coordinates (left, upper, right, lower)
crop_images(folder_path, coordinates, output_folder)
