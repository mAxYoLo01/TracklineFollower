"""From: https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/"""

import cv2

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    im_list =  vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)
    return cv2.resize(im_list, (1600, 900), interpolation=interpolation)

"""From https://math.stackexchange.com/questions/1398634/finding-a-perpendicular-vector-from-a-line-to-a-point"""

def project_point_on_line(slope, intercept, point):
    x = (point[0] + slope * (point[1] - intercept)) / (1 + pow(slope, 2))
    return (x, x * slope + intercept)