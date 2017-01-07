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

#define private public
#include "etl_raw_image.hpp"

using namespace std;
using namespace nervana;

void convert_float_image(cv::Mat mat, const string& path)
{
    float max = 0;
    float* p = (float*)mat.data;
    size_t size = mat.rows * mat.cols;
    for (size_t i=0; i<size; i++)
    {
        max = std::max(max, p[i]);
    }
    for (size_t i=0; i<size; i++)
    {
        p[i] = p[i] / max;
    }
    INFO << max;

    cv::Mat image(mat.rows, mat.cols, CV_8UC1);

    float* s = (float*)mat.data;
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

TEST(raw_image, test)
{
    string source = CURDIR"/test_data/depth/kitchen_0002/r-1294890090.921621-2956011773.bin";
    vector<char> contents = file_util::read_file_contents(source);

    nlohmann::json js = {{"height", 480},
                                {"width", 640},
                                {"debug", true}};
    raw_image::config config{js};
    raw_image::extractor extractor{config};
    std::shared_ptr<image::decoded> dec = extractor.extract(contents.data(), contents.size());

    ASSERT_NE(nullptr, dec);
    ASSERT_EQ(1, dec->get_image_count());

    convert_float_image(dec->get_image(0), "depth.png");
}