#!/usr/bin/env python
import os
print('before iport')
from aeon import DataLoader
print('after iport')

pdir = os.path.dirname(os.path.abspath(__file__))
manifest_root = os.path.join(pdir, '..', '..', 'test', 'test_data')

manifest_file = os.path.join(manifest_root, 'manifest.csv')
cache_root = ""

cfg = {
           'manifest_filename': manifest_file,
           'manifest_root': manifest_root,
           'batch_size': 20,
           'block_size': 40,
           'cache_directory': cache_root,
           'etl': [
               {'type': 'image',
                'channel_major': False,
                'width': 28,
                'height': 28,
                'channels': 1},
               {'type': 'label',
                'binary': False}
           ],
           'augmentation': [
               {'type': 'image',
               'plugin_filename': 'rotate',
               'plugin_params': {"angle": [-45,45]}}
           ]
        }
import time
print("before creation")
d1 = DataLoader(config=cfg)
time.sleep(1)
print("after creation")
print("d1 length {0}".format(len(d1)))

time.sleep(1)
shapes = d1.axes_info
time.sleep(1)
print("shapes: {0}".format(shapes))
time.sleep(1)

for x in d1:
    time.sleep(1)
    image = x[0]
    time.sleep(1)
    label = x[1]
    time.sleep(1)

    print("{0} data: {1}".format(image[0], image[1]))
    time.sleep(1)
    print("{0} data: {1}".format(label[0], label[1]))
    time.sleep(1)

print("finished")
