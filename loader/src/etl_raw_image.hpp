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

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include "interface.hpp"
#include "etl_image.hpp"
#include "image.hpp"
#include "util.hpp"

namespace nervana
{
    namespace raw_image
    {
        class config;
        class extractor;
        class transformer;
        class loader;
    }
}

/**
 * \brief Configuration for image ETL
 *
 * An instantiation of this class controls the ETL of image data into the
 * target memory buffers from the source CPIO archives.
 */
class nervana::raw_image::config : public interface::config
{
public:
    int32_t     height;
    int32_t     width;
    int32_t     channels = 1;
    std::string input_type{"float"};
    std::string output_type{"float"};
    bool        debug = false;

    config(nlohmann::json js);

private:
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(height, mode::REQUIRED),
        ADD_SCALAR(width, mode::REQUIRED),
        ADD_SCALAR(channels, mode::OPTIONAL),
        ADD_SCALAR(input_type, mode::OPTIONAL, [](const std::string& v){ return output_type::is_valid_type(v); }),
        ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v){ return output_type::is_valid_type(v); }),
        ADD_SCALAR(debug, mode::OPTIONAL),
    };

    void validate();
};

class nervana::raw_image::extractor : public interface::extractor<image::decoded>
{
public:
    extractor(const raw_image::config&);
    ~extractor() {}
    virtual std::shared_ptr<image::decoded> extract(const char*, int) override;
private:
    std::string m_input_type;
    int32_t     m_height;
    int32_t     m_width;
    int32_t     m_channels;
};

class nervana::raw_image::loader : public interface::loader<image::decoded>
{
public:
    loader(const raw_image::config& cfg);
    ~loader() {}
    virtual void load(const std::vector<void*>&, std::shared_ptr<image::decoded>) override;

private:
};
