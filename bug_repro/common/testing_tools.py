# INTEL CONFIDENTIAL
# Copyright 2017 Intel Corporation All Rights Reserved.
# The source code contained or described herein and all documents related to the
# source code ("Material") are owned by Intel Corporation or its suppliers or
# licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material may contain trade secrets and proprietary
# and confidential information of Intel Corporation and its suppliers and
# licensors, and is protected by worldwide copyright and trade secret laws and
# treaty provisions. No part of the Material may be used, copied, reproduced,
# modified, published, uploaded, posted, transmitted, distributed, or disclosed
# in any way without Intel's prior express written permission.
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery of
# the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
# Include any supplier copyright notices as supplier requires Intel to use.
# Include supplier trademarks or logos as supplier requires Intel to use,
# preceded by an asterisk. An asterisked footnote can be added as follows:
# *Third Party trademarks are the property of their respective owners.
# Unless otherwise agreed by Intel in writing, you may not remove or alter
# this notice or any other notice embedded in Materials by Intel or Intel's
# suppliers or licensors in any way.


""" Testing tools
    A set of functions used to make functional tests for Aeon augmentations
"""

import numpy as np
from common.testing_logger import log


def compare_colors(p_image, p_color):
    """
    Function check if each pixel at image contains the same pattern color
    :param p_image: image to check
    :param p_color: table [B, G, R] defined pattern color
    :return: True, if image contains only pattern color
             False, if at least one pixel inside the image differs the pattern
    """
    ret_val = True
    shape = p_image.shape
    im_height = shape[1]
    im_width = shape[2]
    for x in range(im_width):
        for y in range(im_height):
            if np.array_equal(p_image[:, y, x], p_color) is not True:
                ret_val = False
                break
        if ret_val is not True:
            break
    return ret_val


def compare_images(p_image_master, p_image_tested):
    """
    The function compares two images based on the difference between them.
    If the average pixel value of difference is greater than 0, then the images differ each other
    :param p_image_master: Master image
    :param p_image_tested: Tested image
    :return: True, if images are the same
             False, if images differ each other
    """
    diff = p_image_master - p_image_tested
    mean = np.mean(diff)
    return False if mean > 0.0 else True




def check_value_in_range(p_val, p_range):
    """
    Function check, if p_val is within range <p_range_min, p_range_max> including limit values
    :param   p_val: value to check
    :param p_range: table [p_range_min, p_range_max]

    :return: True, if p_value is within the range
             False, if p_value is out of range
    """
    return True if ((p_val >= p_range[0]) and (p_val <= p_range[1])) else False


def check_image_gt_boxes(p_images, p_pattern_color):
    """
    Testing image is filled by uniform background color "."
    and contains only one square filled by pattern color "O" at the middle of it.
    Ground truth box "x" indicates exactly that square
    . . . . . . . . . . . . . . . . . . . . . . .          . . . . . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . . . . . .          . . . . . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . . . . . .          . . . . . . . . . . . . . . . . . . . . . . .
    . . . . x x x x x x x x x x x x x x x . . . .          . . . . x x x x x x x x x x x x x x x . . . .
    . . . . x O O O O O O O O O O O O O x . . . .          . . . . x s s s s s s s s s s s s s x . . . .
    . . . . x O O O O O O O O O O O O O x . . . .          . . . . x s O O O O O O O O O O O s x . . . .
    . . . . x O O O O O O O O O O O O O x . . . .          . . . . x s O O O O O O O O O O O s x . . . .
    . . . . x O O O O O O O O O O O O O x . . . .    =>    . . . . x s O O O O O O O O O O O s x . . . .
    . . . . x O O O O O O O O O O O O O x . . . .          . . . . x s O O O O O O O O O O O s x . . . .
    . . . . x O O O O O O O O O O O O O x . . . .          . . . . x s s s s s s s s s s s s s x . . . .
    . . . . x x x x x x x x x x x x x x x . . . .          . . . . x x x x x x x x x x x x x x x . . . .
    . . . . . . . . . . . . . . . . . . . . . . .          . . . . . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . . . . . .          . . . . . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . . . . . .          . . . . . . . . . . . . . . . . . . . . . . .
    The function checks if the ground truth boxes after augmentation process
    continue to indicate the area that contains the blue rectangle.
    It has been noticed that after image processing connected to augmentation,
    color on borders may be slightly blurred "s". Therefore, when checking the color within the ground truth boxes,
    this area is reduced by few pixels each side (safety_collar - s)

    Ground Truth Boxes are normalized at output of Aeon (DataSet),
    so they must be denormalized by multiplying with the height and width of the image.

     :param         p_images: a single batch of Aeon's image set. It is a tuple.
                              p_images[5][1] - array of images
                              p_images[2][1] - count of ground truth boxes
                              p_images[1][1] - array of ground truth boxes [x1, y1, x2, y2]
                                               where x1,y1 define left up corner
                                                     x2,y2 define right bottom corner

     :param  p_pattern_color: table [B, G, R] defined pattern color
     :return: True  - if color inside area indicated by ground truth box (reduced by 2 pixels each side)
                      contains only pattern color
              False - if at least one pixel inside the area differs the pattern

     """
    ret_val = True
    counter = 0
    safety_collar = 3
    for image, boxcount, gt_box_arr in zip(p_images[5][1], p_images[2][1], p_images[1][1]):
        for k in range(int(boxcount)):
            gt_box_denorm = [gt_box_arr[k][0] * (image.shape[2]-1),
                             gt_box_arr[k][1] * (image.shape[1]-1),
                             gt_box_arr[k][2] * (image.shape[2]-1),
                             gt_box_arr[k][3] * (image.shape[1]-1)]
            sub_image = image[
                        :,
                        int(gt_box_denorm[1] + safety_collar): int(gt_box_denorm[3] - safety_collar),
                        int(gt_box_denorm[0] + safety_collar): int(gt_box_denorm[2] - safety_collar)
                        ]
            if compare_colors(sub_image, p_pattern_color):
                log.info('Color comparison inside gt_box (image %d) PASSED' % counter)
            else:
                ret_val = False
                log.error('Color comparison inside gt_box (image %d) FAILED' % counter)
                log.debug('gt_box: LU[%d,%d](+%dpx) -> RB[%d,%d](-%dpx)' %
                          (gt_box_denorm[0], gt_box_denorm[1], safety_collar, 
                           gt_box_denorm[2], gt_box_denorm[3], safety_collar))
        counter += 1
    return ret_val


def check_jaccards(p_images, p_jaccard):
    """
    Function check if after augmentation which give a sample of original image,
    a jaccard factor, of the analyzed sample, is within the assumed range

    Ground Truth Boxes are normalized at output of Aeon (DataSet),
    so they must be denormalized by multiplying with the height and width of the image.

     :param         p_images: a single batch of Aeon's image set. It is a tuple data.
                              p_images[5][1] - array of images
                              p_images[2][1] - count of ground truth boxes
                              p_images[1][1] - array of ground truth boxes [x1, y1, x2, y2]
                                               where x1,y1 define left up corner
                                                     x2,y2 define right bottom corner
    :param p_jaccard: a jaccard range used while augmentation process [jaccard_min, jaccard_max]

    :return: True, if calculated jaccard factor is within assumed range
             False, if jaccard factor doesn't meet mentioned conditions
    """
    ret_val = True
    counter = 0
    for image, boxcount, gt_box_arr in zip(p_images[5][1], p_images[2][1], p_images[1][1]):
        for k in range(int(boxcount)):
            gt_box_denorm = [gt_box_arr[k][0] * (image.shape[2]-1),
                             gt_box_arr[k][1] * (image.shape[1]-1),
                             gt_box_arr[k][2] * (image.shape[2]-1),
                             gt_box_arr[k][3] * (image.shape[1]-1)]
            gb_height = gt_box_denorm[3] - gt_box_denorm[1]
            gb_width = gt_box_denorm[2] - gt_box_denorm[0]
            val_jackard = (gb_height * gb_width) / (image.shape[1] * image.shape[2])

            if check_value_in_range(val_jackard, p_jaccard):
                log.info('The jackard meet the conditions (image %d) PASSED' % counter)
            else:
                ret_val = False
                log.error('The jacquard does not meet the conditions (image %d) FAILED' % counter)
                log.debug('gt_box: LU[%d,%d] -> RB[%d,%d]' %
                          (gt_box_denorm[0], gt_box_denorm[1], gt_box_denorm[2], gt_box_denorm[3]))
                log.debug('gb_height: %d, gb_width: %d' % (gb_height, gb_width))
                log.debug('im_height: %d, im_width: %d' % (image.shape[1], image.shape[2]))
                log.debug('jackard: %f jackard boundaries: [%f, %f]' % (val_jackard, p_jaccard[0], p_jaccard[1]))
        counter += 1
    return ret_val


def check_aspect_ratios(p_images, p_aspect_ratio_range):
    """
    Function check if after augmentation which give a sample of original image,
    a aspect ratio, of the achieved groud truth box, is within the assumed range

    Ground Truth Boxes are normalized at output of Aeon (DataSet),
    so they must be denormalized by multiplying with the height and width of the image.

     :param         p_images: a single batch of Aeon's image set. It is a tuple data.
                              p_images[5][1] - array of images
                              p_images[2][1] - count of ground truth boxes
                              p_images[1][1] - array of ground truth boxes [x1, y1, x2, y2]
                                               where x1,y1 define left up corner
                                                     x2,y2 define right bottom corner
    :param p_aspect_ratio_range: a aspect ratio's range used while augmentation process [asp_ratio_min, asp_ratio_max]

    :return: True, if calculated aspect ratio is within assumed range
            False, if aspect ratio  doesn't meet conditions
    """
    ret_val = True
    counter = 0
    for image, boxcount, gt_box_arr in zip(p_images[5][1], p_images[2][1], p_images[1][1]):
        for k in range(int(boxcount)):
            gb_height = (gt_box_arr[k][3] - gt_box_arr[k][1]) * (image.shape[1]-1)
            gb_width = (gt_box_arr[k][2] - gt_box_arr[k][0]) * (image.shape[2]-1)
            val_aspect_ratio = gb_width / gb_height
            if check_value_in_range(val_aspect_ratio, p_aspect_ratio_range):
                log.info('The aspect ratio meet the conditions (image %d) PASSED' % counter)
            else:
                ret_val = False
                log.error('The aspect ratio does not meet the conditions (image %d) FAILED' % counter)
                log.debug('gt_box: LU[%d,%d] -> RB[%d,%d]' %
                          (gt_box_arr[k][0] * (image.shape[2]-1),      # denormalized
                           gt_box_arr[k][1] * (image.shape[1]-1),      # values
                           gt_box_arr[k][2] * (image.shape[2]-1),      # of coordinates
                           gt_box_arr[k][3] * (image.shape[1]-1)))     # of ground truth boxes

                log.debug('gb_height: %d, gb_width: %d' % (gb_height, gb_width))
                log.debug('aspect ratio: %f range: [%f, %f]' % (val_aspect_ratio,
                                                                p_aspect_ratio_range[0],
                                                                p_aspect_ratio_range[1]))

        counter += 1
    return ret_val
