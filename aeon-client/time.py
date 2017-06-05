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

from data_loader_provider import DataLoaderProvider
from config import Config
from aeon import DataLoader
from utils import generate_manifest
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('--cache', action='store_true', help='Caching on', default=True)
parser.add_argument('-m', '--manifest_file_name', default=None) #'/data/i1k-extracted/train-index.csv')
parser.add_argument('-r', '--manifest_root_path', default='') #'/data/i1k-extracted/')
parser.add_argument('--width', default=256)
parser.add_argument('--height', default=256)
parser.add_argument('--channels', default=3)
parser.add_argument('-c', '--config_path', default=None)
parser.add_argument('--cache_root', default='/tmp')
parser.add_argument('-s', '--seconds', type=int, default=100, help='Timing period')
parser.add_argument('-l', '--manifest_lines', type=int, default=500, help='Lines in randomly generated manifest')
parser.add_argument('-i', '--interval', type=int, default=100, help='Print output interval')

args=parser.parse_args()

if args.cache is not True:
    args.cache_root=''

if args.config_path is not None and args.manifest_path is not None:
    print('Ingoring manifest_path setting and using config_path')

manifest,temp_files=(args.manifest_file_name, []) if args.manifest_file_name\
                                                is not None else\
    generate_manifest(args.manifest_lines, (args.width, args.height, args.channels))
train_config = Config(
    manifest_file=manifest,
    manifest_root=args.manifest_root_path,
    batch_size=args.batch_size,
    image_params=(args.width, args.height, args.channels),
    cache_root=args.cache_root).get()

train_set = DataLoaderProvider.load(train_config)

a = 1
start_time = time.time()
for i in train_set:

    #time.sleep(0.05)
    if (a%args.interval is 0):
        print('Batch read %d' % a)
    a += 1
    if time.time() - start_time > args.seconds:
        break


print('Img/s: %d' % (args.batch_size * a / args.seconds))

for f in temp_files:
    os.remove(f)
