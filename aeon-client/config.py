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
import os

class Config(object):


    def __init__(self,
                manifest_file,
                batch_size,
                #macrobatch=25000,
                cache_root='',
                manifest_root='',
                image_params=(32,32,3),
                label_binary=True,
                data_type='image',
                augmentations=[]):

        self.manifest_file=manifest_file
        self.manifest_root=manifest_root
        self.batch_size=batch_size
        #self.macrobatch=macrobatch
        self.cache_root=cache_root
        self.image_height=image_params[0]
        self.image_width=image_params[1]
        self.image_channels=image_params[2]
        self.label_binary=label_binary
        self.data_type=data_type
        self.augmentations=augmentations


    def get(self):
        config = {'manifest_filename': self.manifest_file,
                  'manifest_root': self.manifest_root,
                  #"iteration_mode": "INFINITE",
                  #"single_thread": self.single_thread,
                  'cache_directory': self.cache_root,
                'etl': ({'type': self.data_type,
                         'height': self.image_height,
                          'width': self.image_width
                         },
                        {
                            'type': 'label',
                            'binary': self.label_binary
                        }),
                'batch_size': self.batch_size}
                #'augmentation': ()}

        for _, aug in enumerate(self.augmentations):
            for key,value in enumerate(aug):
                config[key] = value

        return config
