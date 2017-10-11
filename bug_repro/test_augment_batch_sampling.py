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


""" Testing Aeon augmentation -> Batch Sampling
"""
import argparse
import os

from aeon import DataLoader
from common.testing_tools import check_image_gt_boxes, check_jaccards, check_aspect_ratios
from common.testing_logger import log


def test_augment_batch_sampling(p_batch_size,
                                p_image_height,
                                p_image_width,
                                p_max_sample,
                                p_max_trials,
                                p_sample_jaccard_overlap,
                                p_sampler_aspect_ratio,
                                p_sampler_scale,
                                p_random_seed=0,
                                log_level=0
                                ):
    log.setLevel(log_level)
    log.info('*** Start test_augment_batch_sampling ****************************')
    log.info('** Run test with parameters:\n'
             '\t\t\t\tbatch size: %d\n'
             '\t\t\t\timage [height,width]: [%d,%d]\n'
             '\t\t\t\tmax sample: %d\n'
             '\t\t\t\tmax trials: %d\n'
             '\t\t\t\tsample jaccard overlap: [%.2f,%.2f]\n'
             '\t\t\t\tsample aspect ratio: [%.2f, %.2f]\n'
             '\t\t\t\tscale: [%.2f, %.2f]\n'
             '\t\t\t\trandom seed: %d'
             % (p_batch_size, p_image_height, p_image_width, p_max_sample, p_max_trials,
                p_sample_jaccard_overlap[0], p_sample_jaccard_overlap[1],
                p_sampler_aspect_ratio[0], p_sampler_aspect_ratio[1],
                p_sampler_scale[0], p_sampler_scale[1],
                p_random_seed))
    test_file_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = test_file_dir + '/test_augmentation_data/'

    manifest_filename = 'manifest_square.tsv'

    image_config = {
        'type': 'image',
        'height': p_image_height,
        'width': p_image_width
    }

    localization_config = {
        'type': 'localization_ssd',
        'height': p_image_height,
        'width': p_image_width,
        'class_names': ['square'],
    }

    batch_sampler = {
        'max_trials': p_max_trials,
        'max_sample': p_max_sample,
        'sample_constraint': {
            'max_jaccard_overlap': p_sample_jaccard_overlap[1],
            'min_jaccard_overlap': p_sample_jaccard_overlap[0]
        },
        'sampler': {
            'aspect_ratio': p_sampler_aspect_ratio,
            'scale': p_sampler_scale
        }
    }

    augmentation_config = {
        'type': 'image',
        'crop_enable': False,
        'batch_samplers': [batch_sampler]
    }

    aeon_config = {
        'random_seed': p_random_seed,
        'batch_size': p_batch_size,
        'manifest_root': test_dir,
        'manifest_filename': test_dir + manifest_filename,
        'etl': [localization_config, image_config],  # localization_config must be first.
        'augmentation': [augmentation_config]
    }

    image_set = DataLoader(aeon_config)
    images = next(image_set)
    rv_check_color = check_image_gt_boxes(images, p_pattern_color=[255, 0, 0])
    rv_check_jaccards = check_jaccards(images, p_sample_jaccard_overlap)
    rv_check_aspect_ratios = check_aspect_ratios(images, p_sampler_aspect_ratio)

    ret_val = rv_check_color and rv_check_jaccards and rv_check_aspect_ratios

    if ret_val:
        log.info('PASS: test_augment_batch_sampling')
    else:
        log.error('FAIL: test_augment_batch_sampling')

    log.info('--- Stop test_augment_batch_sampling ------------------------------')
    return ret_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Test Aeon augmentation -> Batch Sampling")

    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed for deterministic mode. 0 (zero) value means non deterministic mode')

    parser.add_argument('--batch_size', type=int, default=1, help='How many images produce during augmentation')

    parser.add_argument('--image_height', type=int, default=300,
                        help='Height of output images (can differ than input ones)')
    parser.add_argument('--image_width', type=int, default=300,
                        help='Width of output images (can differ than input ones)')

    parser.add_argument('--max_sample', type=int, default=1,
                        help='If provided, break when found certain number of samples satisfying the sample_constraint')
    parser.add_argument('--max_trials', type=int, default=50,
                        help='Maximum number of trials for sampling to avoid infinite loop')

    parser.add_argument('--sample_jaccard_overlap_min', type=float, default=0.1,
                        help='minimal value of sample\'s jaccard overlap')
    parser.add_argument('--sample_jaccard_overlap_max', type=float, default=1.0,
                        help='maximum value of sample\'s jaccard overlap')

    parser.add_argument('--sampler_aspect_ratio_min', type=float, default=0.5,
                        help='minimal value of sampler\'s aspect ratio')
    parser.add_argument('--sampler_aspect_ratio_max', type=float, default=2.0,
                        help='maximum value of sampler\'s aspect ratio')

    parser.add_argument('--sampler_scale_min', type=float, default=0.3, help='minimal value of sampler\'s scale')
    parser.add_argument('--sampler_scale_max', type=float, default=1.0, help='maximum value of sampler\'s scale')

    args = parser.parse_args()

    test_augment_batch_sampling(args.batch_size,
                                args.image_height,
                                args.image_width,
                                args.max_sample,
                                args.max_trials,
                                [args.sample_jaccard_overlap_min, args.sample_jaccard_overlap_max],
                                [args.sampler_aspect_ratio_min, args.sampler_aspect_ratio_max],
                                [args.sampler_scale_min, args.sampler_scale_max],
                                args.random_seed)
