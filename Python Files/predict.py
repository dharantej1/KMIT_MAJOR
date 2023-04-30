import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from enhancer_utils import process_low_light_image

# Define the input image path and the model file path
input_image_path = "Python Files/archive/images/images/architecture/╡bside_Duomo_01.jpg"
model_file_path = "Python Files/model_enhancer.h5"

# Process the low-light image using the saved model
enhanced_image = process_low_light_image(input_image_path, model_file_path)
a=np.array(enhanced_image)
a1=Image.fromarray(a)
a1.save('final_enhanced_image.jpg')


# Display the enhanced image
plt.imshow(enhanced_image)
plt.show()
