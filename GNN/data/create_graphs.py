"""
THIS code need an urgent new implementation. This should be done in GPU with some libraries.
Fast implementation about Line of sight and some improvements.
"""

import glob, os, copy, pickle, time, sys
from typing import IO
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cv2
from tqdm import tqdm
import argparse

try:
    from data.page import TablePAGE
except:
    from page import TablePAGE


verify_str_args = ["true", "si"]
info_attributes = False

THRESHOLD = 1
#ROWS
# MAX_WIDTH_DISTANCE = 0.8
# MAX_HEIGHT_DISTANCE = 0.3
#COLS
MAX_WIDTH_DISTANCE = 1
MAX_HEIGHT_DISTANCE = 1
MAX_WIDTH_DISTANCE_noTH = 1
MAX_HEIGHT_DISTANCE_noTH = 1
DEBUG = False
# weights = [10,1] # used for radius
minWidthNumCells=700
sides_per_idx = None

# min_cell = 3
# min_num_neighbours = 6
# mult_punct_h = 1
# mult_punct_w = 2

def get_all_xml(path, ext="xml"):
    # file_names = glob.glob("{}*.{}".format(path,ext))
    file_names = glob.glob(os.path.join(path, f'*{ext}'))
    # print(file_names)
    return file_names

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def bl_to_the_right(n1,n2):
    """
    Return true if coords2 are to the right of coords1
    :param coords1:
    :param coords2:
    :return:
    """
    overlap = 0
    distance = 9999999
    (x1,min_y1),(x1,max_y1) = sides_per_idx[n1]['right']
    (x2,min_y2),(x2,max_y2) = sides_per_idx[n2]['right']
    if x1 < x2: # To the right
        distance = x2 - x1
        if min_y2 >= max_y1 or max_y2 <= min_y1:
            return 0, distance # non overlap
        if max_y2 >= max_y1:
            if min_y2 <= min_y1: # n2 cover all n1
                return max_y1 - min_y1, distance # all n1
            else:
                return max_y1 - min_y2, distance # n2 not cover all n1
        else: # max_y1 is bigger than max_y2
            if min_y1 >= min_y2:
                # return min_y1 - max_y2, distance # n2 not cover all n1
                return max_y2 - min_y1, distance # n2 not cover all n1
            else: # n2 is inside n1
                return max_y2 - min_y2, distance
    return overlap, distance

def bl_to_the_left(n1,n2):
    """
    Return true if coords2 are to the left of coords1
    :param coords1:
    :param coords2:
    :return:
    """
    overlap = 0
    distance = 9999999
    (x1, min_y1), (x1, max_y1) = sides_per_idx[n1]['left']
    (x2, min_y2), (x2, max_y2) = sides_per_idx[n2]['left']
    if x1 > x2:  # To the right
        distance = x1 - x2
        if min_y2 >= max_y1 or max_y2 <= min_y1:
            return 0, distance  # non overlap
        if max_y2 >= max_y1:
            if min_y2 <= min_y1:  # n2 cover all n1
                return max_y1 - min_y1, distance  # all n1
            else:
                return max_y1 - min_y2, distance  # n2 not cover all n1
        else:  # max_y1 is bigger than max_y2
            if min_y1 >= min_y2:
                return min_y1 - max_y2, distance  # n2 not cover all n1
            else:  # n2 is inside n1
                return max_y2 - min_y2, distance
    return overlap, distance

def bl_to_the_upper(n1,n2):
    """
    Return true if coords2 are to the upper of coords1
    :param coords1:
    :param coords2:
    :return:
    """
    (min_x1, y1), (max_x1, y1) = sides_per_idx[n1]['upper']
    (min_x2, y2), (max_x2, y2) = sides_per_idx[n2]['upper']
    overlap = 0
    distance = 9999999
    if y1 > y2:
        distance = y1 - y2
        if min_x1 >= max_x2 or max_x1 <= min_x2:
            return 0, distance  # non overlap
        if max_x2 >= max_x1:
            if min_x2 <= min_x1:  # n2 cover all n1
                return max_x1 - min_x1, distance  # all n1
            else:
                return max_x1 - min_x2, distance  # n2 not cover all n1
        else:  # max_y1 is bigger than max_y2
            if min_x1 >= min_x2:
                return min_x1 - max_x2, distance  # n2 not cover all n1
            else:  # n2 is inside n1
                return max_x2 - min_x2, distance
    return overlap, distance

def bl_to_the_bottom(n1,n2):
    """
    Return true if coords2 are to the bottom of coords1
    :param coords1:
    :param coords2:
    :return:
    """
    (min_x1, y1), (max_x1, y1) = sides_per_idx[n1]['bottom']
    (min_x2, y2), (max_x2, y2) = sides_per_idx[n2]['bottom']
    overlap = 0
    distance = y2 - y1
    if y1 < y2:
        if min_x1 >= max_x2 or max_x1 <= min_x2:
            return 0, distance  # non overlap
        if max_x2 >= max_x1:
            if min_x2 <= min_x1:  # n2 cover all n1
                return max_x1 - min_x1, distance  # all n1
            else:
                return max_x1 - min_x2, distance  # n2 not cover all n1
        else:  # max_y1 is bigger than max_y2
            if min_x1 >= min_x2:
                return max_x2 - min_x1, distance  # n2 not cover all n1
            else:  # n2 is inside n1
                return max_x2 - min_x2, distance

    return overlap, distance

def overlap_same_axis(coords1, coords2):
    """
    Return # if the 2 boundingboxes interesct on X or Y
    :param coord1: BB. EX: [(4223, 3395), (4351, 3391), (4352, 3441), (4224, 3445)]
    :param coords2: BB
    :return:
    """

    maxX1, minX1 = max([i[0] for i in coords1]), min([i[0] for i in coords1])
    maxX2, minX2 = max([i[0] for i in coords2]), min([i[0] for i in coords2])
    maxY1, minY1 = max([i[1] for i in coords1]), min([i[1] for i in coords1])
    maxY2, minY2 = max([i[1] for i in coords2]), min([i[1] for i in coords2])
    overlap = 0
    if (maxX1 > minX2 and maxX2 > minX1): # TODO reformat...
        # length of the second line. L2 is inside L1
        overlap += (maxX2 - minX2)
    elif (maxX2 > minX1 and maxX1 > minX2):
        # L1 inside L2
        overlap += (maxX1 - minX1)
    elif minX1 < maxX2 and maxX2 < maxX1:
        overlap += maxX2 - minX1
    elif maxX1 > minX2 and minX1 < minX2:
        overlap += maxX1 - minX2
    # AUX = overlap
    if (maxY1 > minY2 and maxY2 > minY1):
        # length of the second line. L2 is inside L1
        overlap += (maxY2 - minY2)
    elif (maxY2 > minY1 and maxY1 > minY2):
        # L1 inside L2
        overlap += (maxY1 - minY1)
    elif minY1 < maxY2 and maxY2 < maxY1:
        overlap += maxY2 - minY1
    elif maxY1 > minY2 and minY1 < minY2:
        overlap += maxY1 - minY2
    return overlap

def get_axis_from_points(coords_, n=2, axis=1, func="max"):
    """

    :param coords:
    :param n: number of points to  get
    :param axis: y=1, x=0
    :return:
    """
    points = []
    coords = copy.copy(coords_)
    if type(coords[0]) == list:
        for i in range(n):
            if func == "max":
                m = np.argmax([x[axis] for x in coords])
            else:
                m = np.argmin([x[axis] for x in coords])
            points.append(coords[m])
            # print(coords)
            # print(m)
            del coords[m]
    else:

        aux = [[x[0],x[1]] for x in coords]
        for i in range(n):
            if func == "max":
                m = np.argmax([x[axis] for x in aux])
            else:
                m = np.argmin([x[axis] for x in aux])
            points.append(aux[m])
            del aux[m]

    return points

def get_side_BB(bl, side, width, height):
    min_x = get_axis_from_points(bl, axis=0, func="min", n=1)[0][0]  #
    min_y = get_axis_from_points(bl, axis=1, func="min", n=1)[0][1]  #
    max_x = get_axis_from_points(bl, axis=0, func="max", n=1)[0][0]  #
    max_y = get_axis_from_points(bl, axis=1, func="max", n=1)[0][1] #
    
    width_bb = max_x - min_x
    height_bb = max_y - min_y
    mid_y = np.mean([max_y, min_y])

    min_x = max(0, min_x-1)
    min_y = max(0, min_y-1)
    max_x = min(width, max_x+1)
    max_y = min(height, max_y+1)
    # exit()
    if side == "left" :
        return [(min_x,min_y),(min_x,max_y)]
    elif side == "right":
        return [(max_x,min_y),(max_x,max_y)]
    elif side == "upper": # Upper - lower
        return [(min_x, min_y), (max_x, min_y)]
    elif side == "bottom":
        return [(min_x, max_y), (max_x, max_y)]

def point_in_line(point, line, axis=0):
    """
    Is this point between the points in the line?
    True/False
    :param point: (x,y)
    :param line: [(x',y'), (x'',y'')]
    :param axisx: 0 if its 'x' and 1 if its 'y'
    :return: True or False
    """
    if axis == 1:
        axis_2 = 0
    else:
        axis_2 = 1
    return point[axis_2] > min([line[0][axis_2], line[1][axis_2]]) and \
           point[axis_2] < max([line[0][axis_2], line[1][axis_2]]) and \
           point[axis] == min([line[0][axis], line[1][axis]])

# @timing
def get_neighborhood(triple_bl, pos, BLs, width, height, relations, args, img=""):
    """
    Distance
    :param bb:
    :param BBs:
    :param width:
    :param height:
    :param img:
    :return:
    """
    # BLs[pos] = [id, [(0,0),(0,0)]]
    word_node, prob_node, bl = triple_bl
    # BLs[pos] = (word_node, prob_node, [(0,0),(0,0)])
    max_width_distance = int(width * MAX_WIDTH_DISTANCE)
    max_height_distance = int(height * MAX_HEIGHT_DISTANCE)
    right = sides_per_idx[pos]['right']
    left = sides_per_idx[pos]['left']
    upper = sides_per_idx[pos]['upper']
    bottom = sides_per_idx[pos]['bottom']
    max_y = left[1][1]
    min_y = left[0][1]
    w_box = abs(right[0][0] - left[0][0])
    h_box = abs(upper[0][1] - bottom[0][1])
    punctuation = {}
    neighborhood = []
    neighborhood_feats = []
    BLs_right = []
    # for count, (word, prob, bl_enum) in enumerate(BLs):
    prob = 1
    for count, (bl_enum, word, textLine_id, info) in enumerate(BLs):
        bl_enum = np.array(bl_enum)
        bl_enum = [(min(bl_enum[:,0]), min(bl_enum[:,1]) ), (max(bl_enum[:,0]), max(bl_enum[:,1]))]
        if count == pos: continue
        punct , distance = bl_to_the_right(pos, count)
        # if pos == 26 :
        #     print(punct)
        # if count == 27 and pos == 26:
        #     print(punct , distance)
        if punct >= THRESHOLD and distance <= max_width_distance and  [count, pos] not in relations:  # if its to the right
            BLs_right.append((distance, count, punct, word, prob, bl_enum))
            # BLs_right.append((count, sides_per_idx[count]["left"]))  # We use ONLY the left side of the BB
    BLs_right.sort(key=lambda x: x[0])
    right_punct = right[1][1] - right[0][1]
    right_punct *= args.mult_punct_w # TODO
    for distance, count, punct, word, prob, bl_enum in BLs_right:
        if right_punct > 0:  # we have points to add!
            right_punct -= punct
            punct_count, side, _, _, _ = punctuation.get(count, [0, "right", [], '', 0.0])
            punctuation[count] = [punct_count+punct, side, bl_enum, word, prob]
    # if pos == 26 or pos == 27:
    #     print(pos, punctuation)
    #     exit()
    """END OF right side"""

    """bottom side"""
    BLs_bottom = []
    # for count, (word, prob, bl_enum) in enumerate(BLs):
    for count, (bl_enum, word, textLine_id, info) in enumerate(BLs):
        bl_enum = np.array(bl_enum)
        bl_enum = [(min(bl_enum[:,0]), min(bl_enum[:,1]) ), (max(bl_enum[:,0]), max(bl_enum[:,1]))]
        _, _, w_blenum, _ = geometric_BB(bl_enum, width, height)
        if w_blenum >= args.max_width_line:
            continue
        if count == pos: continue
        punct, distance = bl_to_the_bottom(pos, count)
        if punct >= THRESHOLD and distance <= max_height_distance and  [count, pos] not in relations:
            BLs_bottom.append((distance, count, punct, word, prob, bl_enum))
    BLs_bottom.sort(key=lambda x: x[0])
    bottom_punct = bottom[1][0] - bottom[0][0]
   
    # mid_y = np.mean([max_y, min_y])
    # upper_image = height*bound
    # lower_image = height*(1-bound)
    # # print(mid_y, width, height, upper_image, lower_image)
    # if mid_y <= upper_image or mid_y >= lower_image:
    #     bottom_punct *= mult_punct_h
    bottom_punct *= args.mult_punct_h
    # if pos == 110:
    #     print(punctuation)
    for distance, count, punct, word, prob, bl_enum in BLs_bottom:
        # if pos == 238:
        #     print(pos, " -> ", count, punct, distance, "- bottom_punct {}".format(bottom_punct), bl_enum, -punct)
        if bottom_punct > 0:  # we have points to add!
            bottom_punct -= punct
            punct_count, side, _, _, _ = punctuation.get(count, [0, "bottom", [], '', 0.0])
            punctuation[count] = [punct_count + punct, side, bl_enum, word, prob]
    """END OF up side"""
    # if pos == 110:
    #     print(BLs_bottom)
    #     print(bottom_punct)
    #     print(punctuation)

    count_w = 0
    count_h = 0
    overlap_h = 0
    overlap_w = 0
    ids = []


    y_coord_o = ((upper[0][1] + bottom[0][1]) / 2) / height
    x_coord_o = ((right[0][0] + left[0][0]) / 2) / width
    y_coord_o_topleft = upper[0][1] / height
    x_coord_o_topleft = left[0][0] / width

    for id, (punct, side, points, word, prob) in punctuation.items():
        # punct, side, points, word, prob = punctuation.get(id, [0, "bottom", [], '', 0.0])

        ids.append(id)
        feats = []
        # neighborhood.append((id, bl_enum))
        # Target
        y_coord_t = ((points[0][1] + points[1][1])/2) / height
        x_coord_t = ((points[0][0] + points[1][0])/2) / width
        y_coord_t_topleft = points[0][1] / height
        x_coord_t_topleft = points[0][0] / width
        info = {
            'origin': pos,
            'target': id
        }
        y_coord_mid = (y_coord_o + y_coord_t) / 2
        x_coord_mid = (x_coord_o + x_coord_t) / 2
        y_coord_mid_topleft = (y_coord_o_topleft + y_coord_t_topleft) / 2
        x_coord_mid_topleft = (x_coord_o_topleft + x_coord_t_topleft) / 2
        length_x = abs(x_coord_t - x_coord_o)
        length_y = abs(y_coord_o - y_coord_t)
        length_x_topleft = abs(x_coord_t_topleft - x_coord_o_topleft)
        length_y_topleft = abs(y_coord_o_topleft - y_coord_t_topleft)

        if side == "right":
            overlap_w += punct
            count_w += 1
            punct /= h_box

            # Len / distance
            # length = abs(x_coord_t - x_coord_o)

            # Direction
            feats = [length_x, length_y,
                     length_x_topleft, length_y_topleft,
                     punct,
                     x_coord_o, y_coord_o,
                     x_coord_o_topleft, y_coord_o_topleft,
                     x_coord_t, y_coord_t,
                     x_coord_t_topleft, y_coord_t_topleft,
                     x_coord_mid, y_coord_mid,
                     x_coord_mid_topleft, y_coord_mid_topleft,
                     1]
        elif side == "bottom":
            overlap_h += punct
            count_h += 1
            punct /= w_box


            # Len / distance
            # length = abs(y_coord_o - y_coord_t)

            # Direction
            feats = [length_x, length_y,
                     length_x_topleft, length_y_topleft,
                     punct,
                     x_coord_o, y_coord_o,
                     x_coord_o_topleft, y_coord_o_topleft,
                     x_coord_t, y_coord_t,
                     x_coord_t_topleft, y_coord_t_topleft,
                     x_coord_mid, y_coord_mid,
                     x_coord_mid_topleft, y_coord_mid_topleft,
                     0]
        # print(feats)

        feats.append(info)
        neighborhood_feats.append(feats)
    # normalize overlaps
    overlap_w /= width
    overlap_h /= height
    """Return the ids to link"""
    # print(punctuation)
    # return [id for id,_ in BLs if punctuation.get(id, 0) >= min_thresoold]

        # print(punctuation)
    # if pos in [238,294]:
    #     print(pos, ids)
    #     exit()
    # if pos == 6:
    #     print(ids)
    return ids, neighborhood_feats, count_h, count_w, \
           overlap_w, overlap_h



def calc_distance(coords, coords2, args):
    # print(coords, coords2)
    x1 = np.mean([x[0] for x in coords])
    x2 = np.mean([x[0] for x in coords2])
    y1 = np.mean([x[1] for x in coords])
    y2 = np.mean([x[1] for x in coords2])
    dst = (((x2 - x1)*args.weight_radius_w)**2 + ((y2 - y1)*args.weight_radius_h)**2)**0.5
    # dst2 = (((x2 - x1))**2 + ((y2 - y1))**2)**0.5
    # print([x1,y2], [x2,y2], dst, dst2)
    return dst

# @timing
def get_neighborhood_radius(triple_bl, pos, BLs, width, height, relations, min_radio, args, img=""):
    """
    Distance
    :param bb:
    :param BBs:
    :param width:
    :param height:
    :param img:
    :return:
    """
    # BLs[pos] = [id, [(0,0),(0,0)]]
    word_node, prob_node, bl = triple_bl
    # BLs[pos] = (word_node, prob_node, [(0,0),(0,0)])
    max_dist = min_radio * max(width, height)
    max_width_distance = int(width * MAX_WIDTH_DISTANCE_noTH)
    max_height_distance = int(height * MAX_HEIGHT_DISTANCE_noTH)
    # print(max_width_distance, max_height_distance)
    right = sides_per_idx[pos]['right']
    left = sides_per_idx[pos]['left']
    upper = sides_per_idx[pos]['upper']
    bottom = sides_per_idx[pos]['bottom']
    w_box = abs(right[0][0] - left[0][0])
    h_box = abs(upper[0][1] - bottom[0][1])
    punctuation = {}
    neighborhood = []
    neighborhood_feats = []
    ids = []
    # print(bl)
    # exit()
    
    per_distances = []
    for count, (bl_enum, word, textLine_id, info) in enumerate(BLs):
        if count == pos: continue

        dist = calc_distance(BLs[pos][0], bl_enum, args)
        per_distances.append((count, dist))
    per_distances = sorted(per_distances, key = lambda x: x[1])

    y_coord_o = ((upper[0][1] + bottom[0][1]) / 2) / height
    x_coord_o = ((right[0][0] + left[0][0]) / 2) / width
    y_coord_o_topleft = upper[0][1] / height
    x_coord_o_topleft = left[0][0] / width
    for count, dist in per_distances:
        if dist > max_dist: continue
        if [count, pos] not in relations and [pos, count] not in relations:

            bl_enum, word, textLine_id, info = BLs[count]
            if len(bl_enum) < 2:
                continue
            # Len / distance
            y_coord_t = ((bl_enum[0][1] + bl_enum[1][1])/2) / height
            x_coord_t = ((bl_enum[0][0] + bl_enum[1][0])/2) / width
            y_coord_t_topleft = bl_enum[0][1] / height
            x_coord_t_topleft = bl_enum[0][0] / width
            
            ids.append(count)
            feats = []
            info = {
                'origin': pos,
                'target': count
            }
            y_coord_mid = (y_coord_o + y_coord_t) / 2
            x_coord_mid = (x_coord_o + x_coord_t) / 2
            dist = dist / np.mean([width, height])
            y_coord_mid_topleft = (y_coord_o_topleft + y_coord_t_topleft) / 2
            x_coord_mid_topleft = (x_coord_o_topleft + x_coord_t_topleft) / 2
            length_x = abs(x_coord_t - x_coord_o)
            length_y = abs(y_coord_o - y_coord_t)
            length_x_topleft = abs(x_coord_t_topleft - x_coord_o_topleft)
            length_y_topleft = abs(y_coord_o_topleft - y_coord_t_topleft)
            # Direction
            # feats = [dist, 
            #             0,
            #             x_coord_o, y_coord_o,
            #             x_coord_t, y_coord_t,
            #             x_coord_mid, y_coord_mid,
            #             -1]
            feats = [length_x, length_y,
                     length_x_topleft, length_y_topleft,
                     0, #punct = 0
                     x_coord_o, y_coord_o,
                     x_coord_o_topleft, y_coord_o_topleft,
                     x_coord_t, y_coord_t,
                     x_coord_t_topleft, y_coord_t_topleft,
                     x_coord_mid, y_coord_mid,
                     x_coord_mid_topleft, y_coord_mid_topleft,
                     -1]
            feats.append(info)
            neighborhood_feats.append(feats)
    # exit()
    return ids, neighborhood_feats

# @timing
def calculate_point_sides(idx, width, height):
    global sides_per_idx
    sides_per_idx = {}
    # for i, (word, prob, bb) in enumerate(idx):
    for i, (bl, text, textLine_id, info) in enumerate(idx):
        bl = np.array(bl)
        try:
            bb = [(min(bl[:,0]), min(bl[:,1]) ), (max(bl[:,0]), max(bl[:,1]))]
        except Exception as e:
            print(bl, textLine_id)
            raise e
        sides_per_idx[i] = {
            'right':get_side_BB(bb, 'right', width, height),
            'left':get_side_BB(bb, 'left', width, height),
            'upper':get_side_BB(bb, 'upper', width, height),
            'bottom':get_side_BB(bb, 'bottom', width, height),
            'word':text,
            'used': False,
            'info':info
        }
    return sides_per_idx


def show_all_imgs(drawing, title="", redim=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(1, len(drawing))
    for i in range(len(drawing)):
        # if redim is not None:
        #     a = resize(drawing[i], redim)
        #     print(a.shape)
        #     ax[i].imshow(a)
        # else:
        ax[i].imshow(drawing[i])
    plt.savefig("lines/{}.jpg".format(title), format='jpg', dpi=600)
    # plt.show( dpi=900)
    plt.close()

def save_to_file(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def geometric_BB(bb, width, height):
    y1 = bb[0][1]
    x1 = bb[0][0]
    y2 = bb[1][1]
    x2 = bb[1][0]

    h = abs(y1 - y2)
    w = abs(x1 - x2)
    h = h / height
    w = w / width

    x = ((x1+x2)/2)/width
    y = ((y1+y2)/2)/height
    return x,y,w,h

def get_width_coords(coords):
    min_x = min([x[0] for x in coords])
    max_x = max([x[0] for x in coords])
    return max_x - min_x

def get_max_width_cells(BLs, min_num_cells=2):
    widths = []
    if not BLs:
        return 0
    for i, (coords, c,r) in enumerate(BLs):
        width = get_width_coords(coords)
        widths.append(width)
    maxWidth = max(widths)
    return maxWidth * min_num_cells

def get_neighborhood_node(edges, i):
    list_neighbours = set()
    for s,d in edges:
        if s == i:
            list_neighbours.add(d)
        elif d == i:
            list_neighbours.add(s)
    list_neighbours = list(list_neighbours)
    return list_neighbours

def main(args):
    do_spans = args.do_spans.lower() in verify_str_args
    from_page = args.data_path_tablelabel == ""
    add_by_cell = args.minradio > 0
    reading_order = args.reading_order.lower() in verify_str_args
    prod = args.prod is not None and args.prod.lower() in verify_str_args
    # if add_by_cell:
    #     added__text = "_mincell{}".format(min_cell)
    # else:
    #     added__text = ""
    # corpus="icdar_abp_1098"
    # dir = "/data2/jose/corpus/tablas_DU/{}/".format(corpus)
    # added__text += "_multpunct_w{}_h{}".format(mult_punct_w, mult_punct_h)
    # dir_dest = "/data/READ_ABP_TABLE_ICDAR/{}/all/graphs_structure_BL_maxBLWidth{}{}/".format(corpus, minWidthNumCells, added__text)
    print(args.dir_dest)
    create_dir(args.dir_dest)
    print("Dir dest: ", args.dir_dest)
    fnames = get_all_xml(args.dir)
    graphs = []
    start_all = time.time()
    # for i, x in enumerate(fnames):
    #     print(i, x)
    # exit()
    # fnames = fnames[242:]
    fnames = tqdm(fnames, desc="Files")
    fname_search = "vol003_040_0"
    gts = []
    connections_RO = {}

    for i_fname, fname in enumerate(fnames):
        if "doc" in fname:
            continue
        # if fname_search not in fname:
        #     continue
        print(fname)
        start_page = time.time()
        file_name = fname.split("/")[-1].split(".")[0]
        path_to_save = os.path.join(args.dir_dest, file_name + ".pkl")
        if args.hisclima and not info_attributes and not from_page and not prod:
            IoU = True
            xml_path_gt = os.path.join(args.data_path_tablelabel, fname.split("/")[-1])
            page_table = TablePAGE(xml_path_gt, hisClima=True, prod=prod)
            tls_GT2 = page_table.get_textLinesHisClima() # old
            tls_GT = page_table.get_CellsHisClima()
            tls_GT.extend(tls_GT2)
        else:
            IoU = False
            tls_GT = []
            xml_path_gt = ""
        page = TablePAGE(im_path=fname, hisClima=args.hisclima, info_attributes=False, IoU=IoU, IoU_tl=tls_GT, IoU_path=xml_path_gt, prod=prod)
       
        cells, cell_by_col, cell_by_row = page.get_cells()
        width = page.get_width()
        height = page.get_height()
        if args.hisclima:
            if do_spans:
                if args.headers_file:
                    pass
                    # TODO
                else:
                    # if reading_order:
                    BLs, gts, connections_RO = page.get_TL_headers_RO(info_attributes=info_attributes)
                    # else:
                    #     BLs, gts = page.get_TL_headers()
                    
            else:
                BLs = page.get_textLinesHisClima(info_attributes=info_attributes, maxWidth=int(args.max_width_line * width))
        else:
            BLs = page.get_textLinesFromCell()
        
        ids = []
        ids_tl = []
        nodes = []
        edges = []
        edges_non_repeat = []
        edges_non_repeat_bycell = []
        edge_features = []
        labels = []
        
        info_page = {
            'width': width,
            'height': height,
                }
        calculate_point_sides(BLs, width, height)
        set_ids = set()
        for i, (bl, text, textLine_id, info) in enumerate(BLs):
            if textLine_id in set_ids:
                print(textLine_id, "Repeated")
                continue
            set_ids.add(textLine_id)
            start_line = time.time()

            bl = np.array(bl)
            bb = [(min(bl[:, 0]), min(bl[:, 1])), (max(bl[:, 0]), max(bl[:, 1]))]
            x, y, w, h = geometric_BB(bb, width, height)
            # x, y = bb[0][0], bb[0][1] #punto izq arriba
            y1 = bb[0][1] / height
            x1 = bb[0][0] / width
            y2 = bb[1][1] / height
            x2 = bb[1][0] / width
            labels.append(info)
            

            prob = 1
            word = text
            if w >= args.max_width_line:
                 neighborhood, neighborhood_feats, count_h, count_w, overlap_w, overlap_h = [],[],0,0,0,0
            else:
                neighborhood, neighborhood_feats, count_h, count_w, \
                overlap_w, overlap_h = get_neighborhood((word, prob, bl), i, BLs, width, height, edges_non_repeat, args, img="")

            [edges.append([i, id]) for id in neighborhood]
            # if i in [17,16]:
            #     print(i, neighborhood, edges)
            if not do_spans:
                for id in neighborhood:
                    edges_non_repeat.append([i, id])
                    edges_non_repeat.append([id, i])


            data_node = [
                x1, y1, x2, y2,
                x,y,
                w, h, 
                # prob,
                count_h, count_w,
                overlap_w, overlap_h,
                # fake,
                word, {'node': i, 
                'textLine_id':textLine_id,
                'file_name':file_name
                }
            ]
            nodes.append(data_node)
            # ids.append(textLine_id)
            ids.append("{}_{}".format(fname, i))
            ids_tl.append("{}-{}".format(fname, textLine_id))
            edge_features.extend(neighborhood_feats)
        if add_by_cell:
            
            for i, (bl, text, textLine_id, info) in enumerate(BLs):
                list_neighbours_i = get_neighborhood_node(edges_non_repeat, i)
                if len(list_neighbours_i) > args.min_num_neighbours:
                    continue
                neighborhood_radius, neighborhood_feats_radius = get_neighborhood_radius((word, prob, bl), i, BLs, width, height, edges_non_repeat, args.minradio, args, img="")
                neighborhood.extend(neighborhood_radius)
                # if i in [17,16]:
                #     print(i, "neighborhood_radius", neighborhood_radius)
                # neighborhood_feats.extend(neighborhood_feats_radius)
                edge_features.extend(neighborhood_feats_radius)
                [edges.append([i, id]) for id in neighborhood_radius]
                if not do_spans:
                    for id in neighborhood_radius:
                        edges_non_repeat_bycell.append([i, id])
                        edges_non_repeat_bycell.append([id, i])
                # if i in [17,16]:
                #     print("radius", i, neighborhood_radius, edges)
        # print("edges_non_repeat_bycell", len(edges_non_repeat_bycell))
        edges_non_repeat.extend(edges_non_repeat_bycell)
        if do_spans:
            edges_non_repeat = [] #PARA QUE NO SE ELIMINEN!!! en spans
        new_edges_non_repeated = []
        new_edge_features_non_repeated = []
        set_edges = set()
        for indx, _ in enumerate(zip(edges, edge_features)):
            i,j = edges[indx]
            edge_feature = edge_features[indx]
            if "{}_{}".format(j,i) not in set_edges and not do_spans:
                new_edges_non_repeated.append([i,j])
                new_edge_features_non_repeated.append(edge_feature)
                set_edges.add("{}_{}".format(j,i))
                set_edges.add("{}_{}".format(i,j))
            elif do_spans and "{}_{}".format(i,j) not in set_edges: # añadimos uno
                new_edges_non_repeated.append([i,j])
                new_edge_features_non_repeated.append(edge_feature)
                set_edges.add("{}_{}".format(i,j))
        # if i in [17,16]:
        # print("new_edges_non_repeated ", new_edges_non_repeated)
        # print(set_edges)
        print(" Time for {} pages: {} s with a total of {} edges and {} nodes".format(fname, time.time() - start_page, len(new_edges_non_repeated   ), len(nodes)))
        print(gts)
        # exit()
        data = {
            'fname': fname,
            'nodes': nodes,
            'edges': new_edges_non_repeated,
            'labels': labels,
            'edge_features': new_edge_features_non_repeated,
            'ids': ids,
            'ids_tl': ids_tl,
            'info': info_page,
            'gts_span': gts,
            'connections_RO': connections_RO
        }
        # print(gts)
        save_to_file(data, fname=path_to_save)
    print("Time for all pages: {} s".format( time.time()-start_all))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the graphs')
    parser.add_argument('--do_spans', type=str, help='process only the spans from headers')
    parser.add_argument('--headers_file', type=str, help='The headers file. If its empty the GT will be used')
    
    parser.add_argument('--dir', type=str, help='Path with the PAGE files')
    parser.add_argument('--dir_dest', type=str, help='Path to save the graph files')
    parser.add_argument('--minradio', type=float, help='Minium radio to calculate max distance in case of use distances. max_dist = min_radio * max(width, height)')
    parser.add_argument('--min_num_neighbours', type=int, help='min_num_neighbours')
    parser.add_argument('--mult_punct_w', type=int, help='right_punct *= mult_punct_w')
    parser.add_argument('--mult_punct_h', type=int, help='bottom_punct *= mult_punct_h')
    parser.add_argument('--max_width_line', type=float, help='Will remove the lines smaller than this param. Relative [0,1]')
    parser.add_argument('--weight_radius_w', type=float, help='Used to calculate the distance. Increase width for columns .-> dst = (((x2 - x1)*weight_radius_w)**2 + ((y2 - y1)*weight_radius_h)**2)**0.5')
    parser.add_argument('--weight_radius_h', type=float, help='Used to calculate the distance. Increase height for rows .-> dst = (((x2 - x1)*weight_radius_w)**2 + ((y2 - y1)*weight_radius_h)**2)**0.5')
    parser.add_argument('--hisclima', type=str, help='Special case fo the hisclima corpus')
    parser.add_argument('--data_path_tablelabel', type=str, help='Path to table PAGEs. Used in case to label hyp PAGE from GT PAGE using IoU techniques')
    parser.add_argument('--reading_order', type=str, help='Path to table PAGEs. Used in case to label hyp PAGE from GT PAGE using IoU techniques')
    parser.add_argument('--prod', type=str, help='Path to table PAGEs. Used in case to label hyp PAGE from GT PAGE using IoU techniques')
    args = parser.parse_args()
    main(args)

