import numpy as np
import onnxruntime as ort
from PIL import Image
classes = ["Bird","Flower","Hand","House","Mug","Pencil","Spoon","Sun","Tree","Umbrella"]


ort_session = ort.InferenceSession('model.onnx')  # load the saved onnx model


def Image_crop(image):
    x, y = np.where(image != 0)
    maxX, minX, maxY, minY = max(x), min(x), max(y), min(y);
    return image[minX:maxX, minY:maxY];


def Re_resize(image):
    lenX, lenY = image.shape;
    maxlen = max(lenX, lenY);
    fimage = np.zeros([maxlen, maxlen]);
    y_min = (maxlen - image.shape[0]) // 2;
    y_max = y_min + image.shape[0];
    x_min = (maxlen - image.shape[1]) // 2;
    x_max = x_min + image.shape[1];
    fimage[y_min:y_max, x_min:x_max] = image;
    fimage = Image.fromarray(fimage);
    fimage = fimage.resize([64,64]);
    fimage = (np.array(fimage) > 0.1).astype(np.float32)[None, :, :];
    return fimage;


# pre processing
def process(path):
    # pre process same as training
    image = Image.open(path)
    arr = np.asarray(image)
    image = Image.fromarray(arr[:, :, 3])  # read alpha channel
    image = image.resize((64,64))
    image = (np.array(image) > 0.1).astype(np.float32)
    fimage = Image_crop(image);
    fimage = Re_resize(fimage);

    return fimage[None]


# tes the model
def test(path):
    image = process(path)
    output = ort_session.run(None, {'data': image})[0].argmax()

    print(classes[output], output)

    return classes[output]