import os
from PIL import Image, ImageFilter

import numpy as np
from scipy.ndimage import label

import colorsys

# create log folder if it is not exist
def get_log_folder() -> str:
    dirnow = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(dirnow, "log")
    os.makedirs(logdir, exist_ok=True)
    return logdir

# get a logfile filename
def acquie_log_filename():
    logdir       = get_log_folder()
    logfile_list = list(os.listdir(logdir))
    index = 0
    while ("%07d.png" % index) in logfile_list:
        index += 1
    filename = "%07d.png" % index
    return os.path.join(logdir, filename) # return the smallest number which is

# convert an image into a grey image, and save it into a file
def get_grey_image(input_image: str) -> str:
    img      = Image.open(input_image)
    gray_img = img.convert('L').convert('RGB')
    gray_img.save(logfile:=acquie_log_filename())
    return logfile

# get a gaussian blur for a certain image
def get_blur_image(input_image: str, radius=4) -> str:
    img = Image.open(input_image)
    blurred_img = img.filter(ImageFilter.BoxBlur(radius=radius))
    blurred_img.save(logfile:=acquie_log_filename())
    return logfile

# get a binary image for a certain threshold
def get_bin_image(input_image: str, threshold = 127) -> str:
    img = Image.open(input_image)
    gray_img = img.convert('L')
    binary_img = gray_img.point(lambda p: 255 if p >= threshold else 0)
    binary_img.save(logfile:=acquie_log_filename())
    return logfile

def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

# dump test image
def dump_testing_image_for_cc_detect(labeled:np.ndarray, num_features) -> str:
    assert num_features > 0
    assert len(labeled.shape) == 2 # np.array 2d
    col_len   = 1.0 / num_features
    img       = Image.new('RGB', labeled.shape, 'white')
    for x in range(labeled.shape[0]):
        for y in range(labeled.shape[1]):
            if labeled[x, y] != 0:
                img.putpixel((y, x), hsv_to_rgb(col_len * (labeled[x, y] - 0.5), 0.8, 0.8))
    img.save(logfile:=acquie_log_filename())
    return logfile

# get all connected component in image
def get_cc_for_image(input_image: str):
    image = np.array(Image.open(input_image).convert('L'))
    structure = np.array([ # eight directions is available
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    labeled, num_features = label(image == 0, structure=structure)
    dump_testing_image_for_cc_detect(labeled, num_features)
    return labeled, num_features

# export functions
__all__ = [
    "get_grey_image",
    "get_blur_image",
    "get_bin_image",
    "get_cc_for_image"
]
