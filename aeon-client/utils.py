# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import random
import tempfile
from PIL import Image
import numpy as np
import struct
import json

def load_config(file_path):
    with open(file_path) as config:
        return json.load(config)

def generate_target(filename):
    target = int(random.random() * 1024)
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', target))
    return filename

def generate_image(filename, image_params):
    a = np.random.uniform(-10,10,image_params).astype('uint8')
    img = Image.fromarray(a)
    img.save(filename)

def generate_manifest(num_lines, image_params=(32, 32, 3)):
    temp_files = []
    manifest = tempfile.NamedTemporaryFile(mode='w', delete=False)
    for i in range(num_lines):
        img_filename = tempfile.mkstemp(suffix='.jpg')[1]
        generate_image(img_filename, image_params)
        target_filename = tempfile.mkstemp(suffix='.txt')[1]
        generate_target(target_filename)
        temp_files.append(img_filename)
        temp_files.append(target_filename)
        manifest.write("{}\t{}\n".format(img_filename, target_filename))

    manifest.flush()
    temp_files.append(manifest.name)
    return (manifest.name,temp_files)