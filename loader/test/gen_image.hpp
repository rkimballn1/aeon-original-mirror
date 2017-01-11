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

#include <string>
#include <vector>
#include <memory>
#include <iomanip>

#include "dataset.hpp"
#include "util.hpp"

class gen_image : public dataset<gen_image>
{
public:
    gen_image();
    ~gen_image();

    gen_image& ImageSize( int rows, int cols );

    template<typename T>
    static std::vector<T> render_depth(size_t rows, size_t cols, bool row_major=true)
    {
        std::vector<T> rc;
        for (size_t row=0; row<rows; row++)
        {
            for (size_t col=0; col<cols; col++)
            {
                float value = row * 1000 + col;
                uint8_t tmp[4];
                nervana::pack<uint32_t>((char*)tmp, *(uint32_t*)&value);
                // uint32_t f = *(uint32_t*)tmp;
                // std::cout << "float " << std::hex << std::setw(8) << std::setfill('0') << f << std::endl;
                rc.insert(rc.end(), std::begin(tmp), std::end(tmp));
            }
        }

        return rc;
    }

    std::vector<unsigned char> render_target( int datumNumber ) override;
    std::vector<unsigned char> render_datum( int datumNumber ) override;

    std::vector<unsigned char> RenderImage( int number, int label );

private:
    int         _imageRows;
    int         _imageCols;
};
