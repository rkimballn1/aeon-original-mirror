import os

from aeon import DataLoader
import common.draw_tools as dt


def draw_augmented(aeon_config, output_dir):
    image_set = DataLoader(aeon_config)
    images = next(image_set)
    dt.draw_images(images, _directory=output_dir)


def draw_single_augmented(aeon_config, file_name, output_dir):
    image_set = DataLoader(aeon_config)
    images = next(image_set)
    dt.draw_image(images, _file_name=file_name, _directory=output_dir)

if __name__ == '__main__':

    batch_size = 32
    image_height = 300
    image_width = 300

    test_file_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = test_file_dir + '/sample_images/'
    output_dir = test_file_dir + '/resized/'
    manifest_filename = 'manifest_four_figures_wsta_no_bboxes.tsv'

    image_config = {
        'type': 'image',
        'height': image_height,
        'width': image_width
    }

    localization_config = {
        'type': 'localization_ssd',
        'height': image_height,
        'width': image_width,
        'class_names': ['arrow', 'square', 'triangle', 'wheel'],
    }

    augmentation_config = {
        'type': 'image',
        #'angle': [1, 45],
        'scale': [0.1, 1.0],
        'do_area_scale': True,
        'crop_enable': False
    }

    aeon_config = {
        'batch_size': batch_size,
        'manifest_root': test_dir,
        'manifest_filename': test_dir+manifest_filename,
#        'etl': [localization_config, image_config],  # localization_config must be first.
        'etl': [image_config],
        'augmentation': [augmentation_config]
    }
    draw_augmented(aeon_config, output_dir)
    #draw_single_augmented(aeon_config, 'resized%03d', output_dir)
