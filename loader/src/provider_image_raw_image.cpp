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

#include "provider_image_raw_image.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

extern void convert_float_image(cv::Mat mat, const string& path);
extern void convert_float_image(int height, int width, vector<char>& data, const string& path);

image_raw_image::image_raw_image(nlohmann::json js)
    : m_image_config{js["image"]}
    , m_raw_image_config{js["raw_image"]}
    , m_image_extractor{m_image_config}
    , m_image_transformer{m_image_config}
    , m_image_loader{m_image_config}
    , m_image_factory{m_image_config}
    , m_raw_image_extractor{m_raw_image_config}
    , m_raw_image_loader{m_raw_image_config}
{
    num_inputs = 2;
    oshapes.push_back(m_image_config.get_shape_type());
    oshapes.push_back(m_raw_image_config.get_shape_type());
}

void image_raw_image::provide(int index, buffer_in_array& in_buf, buffer_out_array& out_buf)
{
    std::vector<char>& datum_in  = in_buf[0]->get_item(index);
    std::vector<char>& target_in = in_buf[1]->get_item(index);
    char* datum_out  = out_buf[0]->get_item(index);
    char* target_out = out_buf[1]->get_item(index);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded image with size 0, at index " << index;
        throw std::runtime_error(ss.str());
    }

    // Process image data
    auto image_dec = m_image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params = m_image_factory.make_params(image_dec);
    m_image_loader.load({datum_out}, m_image_transformer.transform(image_params, image_dec));

    // Process target data
    auto blob_dec = m_raw_image_extractor.extract(target_in.data(), target_in.size());
    convert_float_image(blob_dec->get_image(0), "p0.png");

    auto tx = m_image_transformer.transform(image_params, blob_dec);
    convert_float_image(tx->get_image(0), "p1.png");
    m_raw_image_loader.load({target_out}, tx); // should work on tx, not blob_dec

    cv::Mat m(480, 640, CV_32FC1, target_out);
    convert_float_image(m, "p2.png");
}
