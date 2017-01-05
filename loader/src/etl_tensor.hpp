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

#include "interface.hpp"
#include "util.hpp"

namespace nervana
{
    namespace tensor
    {
        class config;
        class decoded;
        class extractor;
        class loader;
    }
}

class nervana::tensor::config : public interface::config
{
public:
    std::string output_type{"float"};
    size_t      output_count;

    config(nlohmann::json js)
    {
        for(auto& info : config_list) {
            info->parse(js);
        }
        verify_config("tensor", config_list, js);

        add_shape_type({output_count}, output_type);
    }

private:
    config()
    {
    }
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v){ return output_type::is_valid_type(v); })
    };
};

class nervana::tensor::decoded : public interface::decoded_media
{
    friend class loader;
public:
    decoded(const char* buf, int bufSize);

    virtual ~decoded() override
    {
    }

private:
    cv::Mat m_mat;
    uint16_t m_magic = 0x4E56;
};


class nervana::tensor::extractor : public interface::extractor<tensor::decoded>
{
public:
    extractor(const tensor::config& cfg);

    ~extractor()
    {
    }

    std::shared_ptr<tensor::decoded> extract(const char* buf, int bufSize) override;
};

class nervana::tensor::loader : public interface::loader<tensor::decoded>
{
public:
    loader(const tensor::config& cfg)
    {
    }

    ~loader()
    {
    }

    void load(const std::vector<void*>& buflist, std::shared_ptr<tensor::decoded> mp) override
    {
    }

private:
};
