import os
from . import image_utils
from . import test_utils

# read an image and return it's pd notation
def link_parse(input_image_path: str) -> list:
    if not os.path.isfile(input_image_path):
        raise FileNotFoundError()
    
    grey_image = image_utils.get_grey_image(input_image_path)
    blur_image = image_utils.get_blur_image(grey_image)
    bina_image = image_utils.get_bin_image(blur_image)
    cc, labeled, numcc  = image_utils.get_cc_for_image(bina_image)
    ep_image, endpoints = image_utils.get_endpoint_for_labeled_image(cc, labeled, numcc)
    match_matrix        = image_utils.get_match_matrix(bina_image, numcc)
    ep_link, mathced_endpoints = image_utils.get_matched_endpoints(cc, match_matrix, endpoints)
    cp_image, cross_pnts       = image_utils.get_crossing_pos(ep_link, ep_image, labeled, numcc)

    return [[1, 5, 2, 4], [3, 1, 4, 6], [5, 3, 6, 2]] # stub