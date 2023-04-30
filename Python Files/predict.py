import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
from enhancer_utils import process_low_light_image

# Define the input image path and the model file path
input_image_path = "Python Files/archive/images/images/architecture/╡bside_Duomo_01.jpg"
model_file_path = "Python Files/model_enhancer.h5"

# Process the low-light image using the saved model
enhanced_image = process_low_light_image(input_image_path, model_file_path)

# Normalize the enhanced image
enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

# Save the enhanced image in JPEG format
output_image_path = "Python Files/enhanced_image.jpg"
enhanced_image_bgr = cv.cvtColor(enhanced_image, cv.COLOR_RGB2BGR)
cv.imwrite(output_image_path, enhanced_image_bgr)

# Display the enhanced image
plt.imshow(enhanced_image)
plt.show()
