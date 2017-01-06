/*
 Copyright 2017 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "etl_raw_image.hpp"

using namespace std;
using namespace nervana;

void raw_image::config::validate()
{
}

raw_image::extractor::extractor(const image::config&)
{
}

shared_ptr<image::decoded> raw_image::extractor::extract(const char*, int)
{

}

raw_image::loader::loader(const raw_image::config& cfg)
{

}

void raw_image::loader::load(const std::vector<void*>&, std::shared_ptr<image::decoded>)
{

}
