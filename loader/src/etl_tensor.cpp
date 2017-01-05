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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_tensor.hpp"
#include "bstream.hpp"
#include "log.hpp"
#include "file_util.hpp"

using namespace std;
using namespace nervana;

tensor::decoded::decoded(const char* inbuf, int insize)
{
    INFO << "decode data";

    dump(cout, inbuf, 128);

    // float max = 0;
    // float* p = (float*)inbuf;
    // size_t size = insize/4;
    // for (size_t i=0; i<size; i++)
    // {
    //     max = std::max(max, p[i]);
    // }
    // cout << __FILE__ << " " << __LINE__ << " " << max << endl;
    // for (size_t i=0; i<size; i++)
    // {
    //     p[i] = p[i] / max;
    // }



    // cout << __FILE__ << " " << __LINE__ << " " << endl;
    // cv::Mat mat(480, 640, CV_32FC1, (void*)inbuf, insize);
    // cv::Mat image(480, 640, CV_8UC1);
    // cout << __FILE__ << " " << __LINE__ << " " << endl;

    // float* s = (float*)mat.data;
    // uint8_t* d = image.data;
    // for (size_t row=0; row<image.rows; row++)
    // {
    //     for (size_t col=0; col<image.cols; col++)
    //     {
    //         *d++ = (*s++)*255;
    //     }
    // }

    // cout << __FILE__ << " " << __LINE__ << " " << endl;
    // cv::imwrite("depth.png", image);

    // cout << __FILE__ << " " << __LINE__ << " " << endl;

    bstream_mem bs{inbuf, (size_t)insize};
    uint16_t magic = bs.readU16();
    uint16_t type = bs.readU16();
    uint16_t channels = bs.readU16();
    uint16_t dims = bs.readU16();
    uint16_t rows = bs.readU16();
    uint16_t cols = bs.readU16();

    INFO << channels;
    INFO << dims;
    INFO << rows;
    INFO << cols;

    m_mat = cv::Mat{rows, cols, CV_MAKETYPE(CV_32F, channels)};
    float* p = (float*)m_mat.data;
    for (int i=0; i<rows*cols; i++)
    {
        p[i] = bs.readF32();
    }
}

/* Extract */
tensor::extractor::extractor(const tensor::config& cfg)
{
}

shared_ptr<tensor::decoded> tensor::extractor::extract(const char* inbuf, int insize)
{
    return make_shared<tensor::decoded>(inbuf, insize);
}
