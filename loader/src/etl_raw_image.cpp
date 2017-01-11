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
#include "log.hpp"

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

    shape_t shape;
    if (channel_major) {
        shape = {channels, height, width};
    } else{
        shape = {height, width, channels};
    }
    add_shape_type(shape, output_type);
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

    cv::Mat mat{(int)m_height, (int)m_width, cv_type, (void*)data, (size_t)size};
    rc->add(mat);

    return rc;
}

raw_image::loader::loader(const raw_image::config& cfg)
    : channel_major{cfg.channel_major}
    , stype{cfg.get_shape_type()}
    , channels{cfg.channels}
{
}

void raw_image::loader::load(const std::vector<void*>& outlist, std::shared_ptr<image::decoded> input)
{
    char* outbuf = (char*)outlist[0];
    // TODO: Generalize this to also handle multi_crop case
    auto cv_type = stype.get_otype().cv_type;
    auto element_size = stype.get_otype().size;
    auto img = input->get_image(0);
    int image_size = img.channels() * img.total() * element_size;

    for (int i=0; i < input->get_image_count(); i++)
    {
        auto outbuf_i = outbuf + (i * image_size);
        auto input_image = input->get_image(i);
        vector<cv::Mat> source;
        vector<cv::Mat> target;
        vector<int>     from_to;

        source.push_back(input_image);
        if (channel_major)
        {
            for(int ch=0; ch<channels; ch++)
            {
                target.emplace_back(img.size(), cv_type, (char*)(outbuf_i + ch * img.total() * element_size));
                from_to.push_back(ch);
                from_to.push_back(ch);
            }
        }
        else
        {
            target.emplace_back(input_image.size(), CV_MAKETYPE(cv_type, channels), (char*)(outbuf_i));
            for(int ch=0; ch<channels; ch++)
            {
                from_to.push_back(ch);
                from_to.push_back(ch);
            }
        }
        image::convert_mix_channels(source, target, from_to);
    }
}
