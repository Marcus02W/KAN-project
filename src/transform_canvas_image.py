from PIL import Image, ImageOps
import numpy as np

def transform_canvas_image(drawing):

    img_to_predict = drawing.image_data
    img_to_predict = Image.fromarray(img_to_predict.astype("uint8"))

    #border_size = 10  # You can adjust this value
    #img_to_predict = ImageOps.expand(img_to_predict, border=border_size, fill='white')

    img_to_predict = img_to_predict.resize((8, 8), Image.LANCZOS)
    r, g, b, a = img_to_predict.split()
    img_to_predict = Image.merge("RGB", (r, g, b))
    img_to_predict = img_to_predict.convert("L")
    img_to_predict = ImageOps.invert(img_to_predict)
    img_to_predict = np.array(img_to_predict)
    img_to_predict = img_to_predict / 255.0

    return img_to_predict