import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import math
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import colorsys

CLEAN_LOG_FOLDER_WHEN_INIT = True

from .bfs import get_diameter

# create log folder if it is not exist
def get_log_folder() -> str:
    dirnow = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(dirnow, "log")
    os.makedirs(logdir, exist_ok=True)
    return logdir

# remove all cached log file
def clean_log_folder():
    logdir = get_log_folder()
    for file in os.listdir(logdir):
        filepath = os.path.join(logdir, file)
        if os.path.isfile(filepath):
            os.remove(filepath)

if CLEAN_LOG_FOLDER_WHEN_INIT:
    clean_log_folder()

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
    img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(logfile:=acquie_log_filename())
    return logfile

# get all connected component in image
def get_cc_for_image(input_image: str):
    image = np.array(Image.open(input_image).convert('L')).T
    structure = np.array([ # eight directions is available
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    labeled, num_features = label(image == 0, structure=structure)
    cc_image = dump_testing_image_for_cc_detect(labeled, num_features)
    cc_list = []
    for cc_id in range(1, 1 + num_features):
        cc_list.append(np.where(labeled == cc_id)[0])
        print("[knotpen] component size for component %d: %d" % (cc_id, len(cc_list[-1])))
    return cc_image, labeled, num_features

def get_endpoint_for_labeled_image(cc_image, labeled, num_features, radius = 12, color='red', width=3):
    results   = get_diameter(labeled, num_features)
    img       = Image.open(cc_image)
    draw      = ImageDraw.Draw(img)
    endpoints = []
    for _, y, x in results:
        for xy in (x, y):
            left   = xy[0] - radius
            top    = xy[1] - radius
            right  = xy[0] + radius
            bottom = xy[1] + radius
            draw.ellipse([left, top, right, bottom], outline=color, width=width)
            endpoints.append(tuple(int(item) for item in xy)) # record end point position
    img.save(logfile:=acquie_log_filename())
    return logfile, endpoints

# when we join two end point by a line, and the line cross nothing between them, it's not availabel
# useful when processing doubling and cabling
def get_match_matrix(bin_image, num_cc):
    match_matrix = np.ones(shape=(num_cc * 2, num_cc * 2))
    for i in range(num_cc * 2):
        for j in range(num_cc * 2):
            if i >= j:
                match_matrix[i, j] = math.inf
    print("[knotpen] warning: get_match_matrix has not been implemented.")
    return match_matrix

def get_euclid_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) 

def remove_min_row_col(matrix):
    results = []
    row_index = list(range(matrix.shape[0]))
    col_index = list(range(matrix.shape[1]))
    while matrix.size > 0:  # when matrix is not empty
        min_index = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
        min_row, min_col = min_index
        if not math.isinf(matrix[min_row, min_col]):
            results.append((row_index[min_row], col_index[min_col]))
        del row_index[min_row]
        del col_index[min_col]
        matrix = np.delete(matrix, min_row, axis=0)  # delete row
        matrix = np.delete(matrix, min_col, axis=1)  # delete coloumn
    return results

def reverse_tuple(pr2):
    return tuple(list(pr2)[::-1])

def shorten_line_segment(x1, y1, x2, y2, shorten_by):
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length <= 2 * shorten_by:
        return (x1, y1), (x2, y2)
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    new_x1 = x1 + dx * shorten_by
    new_y1 = y1 + dy * shorten_by
    new_x2 = x2 - dx * shorten_by
    new_y2 = y2 - dy * shorten_by
    return (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2))

def create_ep_link_image_on_ep_image(ep_image, endpoints, match_pairs, color='black', width=4) -> str:
    img = Image.open(ep_image)
    draw = ImageDraw.Draw(img)
    for id_1, id_2 in match_pairs:
        start_point = endpoints[id_1]
        end_point   = endpoints[id_2]
        start_point, end_point = shorten_line_segment(*start_point, *end_point, 10)
        draw.line([start_point, end_point], fill=color, width=width)
    img.save(logfile:=acquie_log_filename())
    return logfile

# list of tuple
def get_matched_endpoints(cc_image: str, match_matrix: np.ndarray, endpoints: list) -> list:
    distance_matrix = match_matrix.copy()
    for i in range(len(endpoints)):
        for j in range(len(endpoints)):
            if not math.isinf(distance_matrix[i, j]):
                distance_matrix[i, j] = get_euclid_distance(endpoints[i], endpoints[j])
    match_pairs = remove_min_row_col(distance_matrix)
    ep_link = create_ep_link_image_on_ep_image(cc_image, endpoints, match_pairs)
    return ep_link, match_pairs

def find_connected_components_centroid(matrix):
    structure = np.ones((3, 3), dtype=int) 
    labeled_array, num_features = label(matrix, structure)
    components = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        centroid = center_of_mass(matrix, labeled_array, i)
        components.append((coords.tolist(), tuple(int(item) for item in centroid)))
    return components

def dump_crossing_pos_on_ep_link_image(ep_image: str, centroids, radius=12, color='green', width=4):
    img  = Image.open(ep_image).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _, centroid in centroids: # plist is ignored
        left   = centroid[0] - radius
        top    = centroid[1] - radius
        right  = centroid[0] + radius
        bottom = centroid[1] + radius
        draw.ellipse([left, top, right, bottom], outline=color, width=width)
    img.save(logfile:=acquie_log_filename())
    return logfile

def debug_show_matrix_2d(matrix):
    plt.imshow(matrix, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('2D Array Visualization')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def get_crossing_pos(ep_link: str, ep_image: str, labeled: np.ndarray, numcc: int):
    img  = np.array(Image.open(ep_link).convert('L')).T
    mask = np.zeros(labeled.shape)
    debug_show_matrix_2d(img)
    debug_show_matrix_2d(labeled)
    assert reverse_tuple(img.shape) == labeled.shape
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0 and labeled[i, j] != 0:
                mask[i, j] = 1
    debug_show_matrix_2d(mask)
    centroids = find_connected_components_centroid(mask)
    cp_image  = dump_crossing_pos_on_ep_link_image(ep_image, centroids)
    return cp_image, centroids

# export functions
__all__ = [
    "get_grey_image",
    "get_blur_image",
    "get_bin_image",
    "get_cc_for_image",
    "get_crossing_pos"
]
