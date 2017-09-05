import numpy as np
from PIL import Image as PILImage, ImageDraw
import json
import argparse

from aeon import DataLoader
cnt = 0


def draw_images(batch):
    global cnt
    images = batch['image']
    gt_box_count = batch['gt_box_count']
    gt_boxes = batch['gt_boxes']
    image_shape = batch['image_shape']
    difficult_flag = batch['difficult_flag']
    gt_class_count = batch['gt_class_count']
    for image, boxcount, gt_box_arr in zip(images, gt_box_count, gt_boxes):
        img = PILImage.fromarray(np.dstack(image)[:, :, ::-1])
        draw = ImageDraw.Draw(img)
        for k in range(boxcount):
            gt_rectangle = [
                gt_box_arr[k][0] * 300, gt_box_arr[k][1] * 300,
                gt_box_arr[k][2] * 300, gt_box_arr[k][3] * 300
            ]
            draw.rectangle(gt_rectangle)

        img.save('aeon-output%d.png' % cnt)
        cnt += 1


def test_loader_exception_next(num_of_batches_to_process, net_definition_file,
                               manifest_filename, manifest_root):
    config = dict()
    with net_definition_file as cfg_file:
        contents = cfg_file.read()
        config = json.loads(contents)
        config['manifest_filename'] = manifest_filename
        config['manifest_root'] = manifest_root
        del config['cache_directory']

    dl = DataLoader(config)
    for x in range(0, num_of_batches_to_process):
        draw_images(dl.next())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Draw images and their bounding boxes saving them in workdir.")

    parser.add_argument(
        '--num_batches',
        type=int,
        required=True,
        help='number of batches to use')
    parser.add_argument(
        '--config', type=file, required=True, help='ssd config file')
    parser.add_argument(
        '--manifest_filename', type=str, required=True, help='manifest path')
    parser.add_argument(
        '--manifest_root',
        type=str,
        required=True,
        help='manifest root directory')

    args = parser.parse_args()
    num_of_batches_to_process = args.num_batches
    net_definition_file = args.config
    manifest_filename = args.manifest_filename
    manifest_root = args.manifest_root

    test_loader_exception_next(num_of_batches_to_process, net_definition_file,
                               manifest_filename, manifest_root)
    print "%d images drawn" % cnt
