import os
import cv2
import numpy as np

# Function to convert PNG image to TXT
def convert_to_txt(input_file, output_file):
    np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})

    img = cv2.imread(input_file, 0)
    if img.shape != [28, 28]:
        img = cv2.resize(img, (28, 28))
    
    img = img.reshape(28, 28, -1)

    # Revert the image and normalize it to 0-1 range
    img = img / 255.0

    # Save the image in a txt file in row major format
    np.savetxt(output_file, img.flatten(), fmt='%f', delimiter=',')

# Folder containing PNG files
folder_path = 'test'

# Output folder for TXT files
output_folder = 'testtext'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over PNG files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # Construct input and output file paths
        input_file = os.path.join(folder_path, filename)
        output_file = os.path.join(output_folder, filename.replace('.png', '.txt'))

        # Convert PNG to TXT
        convert_to_txt(input_file, output_file)

print("Conversion completed successfully.")
