import cv2 as cv
import numpy as np
from keras.models import load_model

def process_low_light_image(image_path, model_path):
    def noisy(noise_typ, image):
        if noise_typ == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 0.0001
            sigma = var ** 0.05
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = gauss + image
            return noisy
        elif noise_typ == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 1.0
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(image.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
            out[coords] = 1
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy
    
    def ExtractTestInput(ImagePath):
        img = cv.imread(ImagePath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_ = cv.resize(img, (500, 500))
        hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV)  # convert it to hsv
        hsv[..., 2] = hsv[..., 2] * 0.2
        img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        Noise = noisy("s&p", img1)
        Noise = Noise.reshape(1, 500, 500, 3)
        return Noise

    # Load the trained model
    model = load_model(model_path)

    # Process the input image
    img = ExtractTestInput(image_path)

    # Make the prediction
    prediction = model.predict(img)

    # Reshape the prediction to an image
    enhanced_image = prediction.reshape(500, 500, 3)

    return enhanced_image
