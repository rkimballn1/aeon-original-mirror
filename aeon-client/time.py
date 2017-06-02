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
parser.add_argument('-b', '--batch_size', type=int, default=20)
parser.add_argument('-c', '--cache', action='store_true', help='Caching on')
parser.add_argument('--cache_root', default='/tmp')
parser.add_argument('-s', '--seconds', type=int, default=20, help='Timing period')
parser.add_argument('-l', '--manifest_lines', type=int, default=200, help='Lines in randomly generated manifest')
args=parser.parse_args()

if args.cache is not True:
    args.cache_root=''

manifest,temp_files=generate_manifest(args.manifest_lines)
train_config = Config(
    manifest_file=manifest.name,
    batch_size=args.batch_size,
    cache_root=args.cache_root).get()

train_set = DataLoaderProvider.load(train_config)

a = 1
start_time = time.time()
test_miliseconds=args.seconds * 1000
for i in train_set:
    if (a%1000 is 0):
        print('Batch read %d' % a)
    a += 1
    if time.time() - start_time > args.seconds:
        break

print('Img/s: %d' % (args.batch_size * a / args.seconds))

for f in temp_files:
    os.remove(f)
