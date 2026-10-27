import rembg
import numpy as np
from PIL import Image

# Load the input image. The image doesn't support transparency. 
input_image = Image.open('backgrounds/input_image.jpg')

# Convert the input image to a numpy array. The image doesn't support transparency.
# input_array.shape: (1080, 1920, 3) → (Height, Width, [R, G, B]) 
input_array = np.array(input_image)

'''
Apply background removal using rembg.
Image is analyzed using deep learning model to separate the
foreground object from the background 

Adding 4th (Alpha) Channel to support transparency:
- Instead of deleting pixels, rembg.remove expands the image arrary 
- output_array.shape: (1080, 1920, 4) → (Height, Width, [R, G, B, A])
'''
output_array = rembg.remove(input_array)

# Convert from the output array to PIL Image (object vs grid of numbers)
output_image = Image.fromarray(output_array)

# Save the output image. Saving as .jpg won't work. 
output_image.save('output_image.png')