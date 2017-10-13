import numpy as np
import os
from PIL import Image as PILImage, ImageDraw

cnt = 1


def ensure_dir_exists(_dir):
    directory = os.path.dirname(_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)


def draw_images(p_images, _prestring='aeon-output', _directory='output/'):
    global cnt

    for image in p_images[0][1]:
        img = PILImage.fromarray(np.dstack(image)[:, :, ::-1])
        draw = ImageDraw.Draw(img)
#        for k in range(boxcount):
#            gt_rectangle = [
#                gt_box_arr[k][0] * (image.shape[2]-1),
#                gt_box_arr[k][1] * (image.shape[1]-1),
#                gt_box_arr[k][2] * (image.shape[2]-1),
#                gt_box_arr[k][3] * (image.shape[1]-1)
#            ]
#            draw.rectangle(gt_rectangle)
        ensure_dir_exists(_directory)
        img.save('%s%s_%03d.png' % (_directory, _prestring, cnt))
        cnt += 1


def draw_image(_image, _file_name='aeon-output', _directory='output/'):
    img = PILImage.fromarray(np.dstack(_image)[:, :, ::-1])
    ImageDraw.Draw(img)
    ensure_dir_exists(_directory)
    img.save('%s%s.png' % (_directory, _file_name))

