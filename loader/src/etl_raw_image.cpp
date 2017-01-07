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

raw_image::config::config(nlohmann::json js)
{
    if(js.is_null()) {
        throw std::runtime_error("missing raw_image config in json config");
    }

    for(auto& info : config_list) {
        info->parse(js);
    }
    verify_config("image", config_list, js);
}

void raw_image::config::validate()
{
}

raw_image::extractor::extractor(const raw_image::config& config)
    : m_input_type{config.input_type}
    , m_height{config.height}
    , m_width{config.width}
    , m_channels{config.channels}
{
}

shared_ptr<image::decoded> raw_image::extractor::extract(const char* data, int size)
{
    shared_ptr<image::decoded> rc = make_shared<image::decoded>();
    int cv_type = 0;

    if (m_input_type == "int8_t")
    {
        cv_type = CV_MAKETYPE(CV_8S, m_channels);
    }
    else if (m_input_type == "uint8_t")
    {
        cv_type = CV_MAKETYPE(CV_8U, m_channels);
    }
    else if (m_input_type == "int16_t")
    {
        cv_type = CV_MAKETYPE(CV_16S, m_channels);
    }
    else if (m_input_type == "uint16_t")
    {
        cv_type = CV_MAKETYPE(CV_16U, m_channels);
    }
    else if (m_input_type == "int32_t")
    {
        cv_type = CV_MAKETYPE(CV_32S, m_channels);
    }
    else if (m_input_type == "uint32_t")
    {
    }
    else if (m_input_type == "float")
    {
        cv_type = CV_MAKETYPE(CV_32F, m_channels);
    }
    else if (m_input_type == "double")
    {
        cv_type = CV_MAKETYPE(CV_64F, m_channels);
    }
    else if (m_input_type == "char")
    {        
        cv_type = CV_MAKETYPE(CV_8U, m_channels);
    }

    cv::Mat mat{m_height, m_width, cv_type, data, size};

    return rc;
}

raw_image::loader::loader(const raw_image::config& cfg)
{

}

void raw_image::loader::load(const std::vector<void*>&, std::shared_ptr<image::decoded>)
{

}
