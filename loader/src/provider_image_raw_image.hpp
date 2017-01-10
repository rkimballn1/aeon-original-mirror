/*
 Copyright 2016 Nervana Systems Inc.
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

#pragma once

#include "provider_interface.hpp"
#include "etl_raw_image.hpp"
#include "etl_image.hpp"

namespace nervana
{
    class image_raw_image;
}

class nervana::image_raw_image : public provider_interface
{
public:
    image_raw_image(nlohmann::json js);
    void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);

private:
    image::config              m_image_config;
    raw_image::config          m_raw_image_config;
    image::extractor           m_image_extractor;
    image::transformer         m_image_transformer;
    image::loader              m_image_loader;
    image::param_factory       m_image_factory;
    raw_image::extractor       m_raw_image_extractor;
    raw_image::loader          m_raw_image_loader;
};
