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

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"

#include "log.hpp"
#include "file_util.hpp"
#include "provider_factory.hpp"
#include "gen_image.hpp"

#define private public
#include "etl_raw_image.hpp"

using namespace std;
using namespace nervana;

void convert_float_image(cv::Mat mat, const string& path)
{
    float max = 0;
    const float* p = (const float*)mat.data;
    size_t size = mat.rows * mat.cols;
    for (size_t i=0; i<size; i++)
    {
        max = std::max(max, p[i]);
    }

    cv::Mat image(mat.rows, mat.cols, CV_8UC1);

    const float* s = (float*)mat.data;
    uint8_t* d = image.data;
    for (size_t row=0; row<image.rows; row++)
    {
        for (size_t col=0; col<image.cols; col++)
        {
            *d++ = (*s++)/max*255;
        }
    }

    cv::imwrite(path, image);
}

void convert_float_image(int height, int width, vector<char>& data, const string& path)
{
    cv::Mat mat(height, width, CV_32FC1, data.data());
    return convert_float_image(mat, path);
}

TEST(raw_image, extractor)
{
    size_t width = 640;
    size_t height = 480;
    vector<char> contents = gen_image::render_depth<char>(height, width);
    nlohmann::json js = {{"height", height},
                                {"width", width},
                                {"debug", true}};
    raw_image::config config{js};
    raw_image::extractor extractor{config};
    std::shared_ptr<image::decoded> dec = extractor.extract(contents.data(), contents.size());

    ASSERT_NE(nullptr, dec);
    ASSERT_EQ(1, dec->get_image_count());

    cv::Mat image = dec->get_image(0);
    convert_float_image(image, "depth_extractor.png");
}

TEST(raw_image, provider)
{
    size_t width = 640;
    size_t height = 480;
    nlohmann::json js =
     {{"type","image,raw_image"},
        {"image", {
            {"height",height},
            {"width",width},
            {"channel_major",false},
            {"flip_enable",true}}},
        {"raw_image", {
            {"height", height},
            {"width", width},
            {"debug", true}
    }}};

    auto media = nervana::provider_factory::create(js);
    ASSERT_NE(nullptr, media);
    const vector<nervana::shape_type>& oshapes = media->get_oshapes();

    ASSERT_NE(0, oshapes.size());
    size_t dsize = oshapes[0].get_byte_size();
    size_t tsize = width * height * sizeof(float);

    size_t batch_size = 128;

    buffer_out_array outBuf({dsize, tsize}, batch_size);

    gen_image gi;
    gi.ImageSize(height, width);
    vector<uint8_t> image_ = gi.render_datum(0);
    vector<char> depth_image = gen_image::render_depth<char>(height, width);

    vector<char> image(image_.begin(), image_.end());
    convert_float_image(height, width, depth_image, "depth0.png");

    buffer_in_array bp(2);
    buffer_in& data_p = *bp[0];
    buffer_in& target_p = *bp[1];
    data_p.add_item(image);
    target_p.add_item(depth_image);

    media->provide(0, bp, outBuf);

    cv::Mat mat(height,width,CV_32FC1,outBuf[1]->get_item(0));
    convert_float_image(mat, "depth_output.png");

    float* f = (float*)(outBuf[1]->get_item(0));
    size_t index = 0;
    for (size_t row=0; row<height; row++)
    {
        for (size_t col=0; col<width; col++)
        {
            size_t c = size_t(f[index]) % 1000;
            size_t r = size_t(f[index]) / 1000;
            ASSERT_EQ(col, c);
            ASSERT_EQ(row, r);
            index++;
        }
    }
}