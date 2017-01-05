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
#include "etl_tensor.hpp"

using namespace std;
using namespace nervana;

void convert_float_image(cv::Mat mat)
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

    cv::imwrite("depth.png", image);
}

TEST(tensor, depthmap)
{
    string test_file = CURDIR"/test_data/test_image.ten";
    // {
    //     uint16_t m_magic = 0x4E56;
    //     string source = CURDIR"/test_data/depth/kitchen_0002/r-1294890090.921621-2956011773.bin";
    //     {
    //         vector<char> contents = file_util::read_file_contents(source);
    //         ofstream f(test_file, ofstream::binary);
    //         if (f)
    //         {
    //             f.write((const char*)&m_magic, sizeof(m_magic));
    //             uint16_t type = 0;
    //             uint16_t channels = 1;
    //             uint16_t dims = 2;
    //             uint16_t rows = 480;
    //             uint16_t cols = 640;
    //             f.write((const char*)&type, 2);
    //             f.write((const char*)&channels, 2);
    //             f.write((const char*)&dims, 2);
    //             f.write((const char*)&rows, 2);
    //             f.write((const char*)&cols, 2);
    //             for (size_t i=0; i<contents.size(); i++)
    //             {
    //                 f << (uint8_t)contents.data()[i];
    //             }
    //         }
    //     }
    //     {
    //         vector<char> tmp = file_util::read_file_contents(test_file);
    //         cout << endl << endl;
    //         dump(cout, tmp.data(), 128);
    //     }
    // }

    vector<char> contents = file_util::read_file_contents(test_file);
    tensor::decoded dec{contents.data(), (int)contents.size()};

    convert_float_image(dec.m_mat);
}