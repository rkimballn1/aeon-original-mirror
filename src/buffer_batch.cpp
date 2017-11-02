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

#include <algorithm>
#include <stdexcept>

#include "buffer_batch.hpp"
#include "log.hpp"
#include <xmmintrin.h>

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

using namespace std;
using namespace nervana;

variable_record_field& encoded_record::element(size_t index)
{
    if (m_elements.size() <= index)
    {
        throw out_of_range("encoded_record element out of range access");
    }
    return m_elements[index];
}

const variable_record_field& encoded_record::element(size_t index) const
{
    if (m_elements.size() <= index)
    {
        throw out_of_range("encoded_record element out of range access");
    }
    return m_elements[index];
}

buffer_fixed_size_elements::buffer_fixed_size_elements(const shape_type& shp_tp,
                                                       size_t            batch_size,
                                                       bool              pinned)
    : m_shape_type{shp_tp}
    , m_size{m_shape_type.get_byte_size() * batch_size}
    , m_batch_size{batch_size}
    , m_stride{m_shape_type.get_byte_size()}
    , m_pinned{pinned}
{
    allocate();
}

buffer_fixed_size_elements::buffer_fixed_size_elements(const buffer_fixed_size_elements& rhs)
    : m_data{nullptr}
    , m_shape_type{rhs.m_shape_type}
    , m_size{rhs.m_size}
    , m_batch_size{rhs.m_batch_size}
    , m_stride{rhs.m_stride}
    , m_pinned{rhs.m_pinned}
{
    allocate();
    memcpy(m_data, rhs.m_data, m_size);
}

char* buffer_fixed_size_elements::get_item(size_t index)
{
    size_t offset = index * m_stride;
    if (index >= (int)m_batch_size)
    {
        throw invalid_argument("buffer_fixed_size: index out-of-range");
    }
    return &m_data[offset];
}

cv::Mat buffer_fixed_size_elements::get_item_as_mat(size_t index, bool channel_major) const
{
    std::vector<int> sizes;
    size_t           channels;
    for (auto& d : m_shape_type.get_shape())
    {
        sizes.push_back(static_cast<int>(d));
    }
    int ndims = static_cast<int>(sizes.size());

    if (channel_major)
    {
        channels = sizes[0];
    }
    else
    {
        ndims -= 1;
        channels = sizes.back();
        sizes.pop_back();
    }

    cv::Mat ret(ndims,
                &sizes[0],
                CV_MAKETYPE(m_shape_type.get_otype().get_cv_type(), channels),
                (void*)&m_data[index * m_stride]);
    return ret;
}

const char* buffer_fixed_size_elements::get_item(size_t index) const
{
    size_t offset = index * m_stride;
    if (index >= (int)m_batch_size)
    {
        throw invalid_argument("buffer_fixed_size: index out-of-range");
    }
    return &m_data[offset];
}

void buffer_fixed_size_elements::allocate()
{
#if HAS_GPU
    if (m_pinned)
    {
        CUresult status = cuMemAllocHost((void**)&m_data, m_size);
        if (status != CUDA_SUCCESS)
        {
            throw std::bad_alloc();
        }
    }
    else
    {
        m_data = new char[m_size];
    }
#else
    m_data = new char[m_size];
#endif
}

buffer_fixed_size_elements::~buffer_fixed_size_elements()
{
#if HAS_GPU
    if (m_pinned)
    {
        cuMemFreeHost(m_data);
    }
    else
    {
        delete[] m_data;
    }
#else
    delete[] m_data;
#endif
}

void fixed_buffer_map::copy(fixed_buffer_map& src, size_t src_index, size_t dst_index, size_t count)
{
    for (auto name : m_names)
    {
        buffer_fixed_size_elements* src_fbm = src[name];
        buffer_fixed_size_elements* dst_fbm = operator[](name);
        char*                       p_src   = src_fbm->get_item(src_index);
        char*                       p_dst   = dst_fbm->get_item(dst_index);

        if ((count + src_index > src_fbm->get_item_count()) ||
            (count + dst_index > dst_fbm->get_item_count()))
            throw invalid_argument("buffer_fixed_size: count out-of-range");

        memcpy(p_dst, p_src, count * src_fbm->get_stride());
    }
}

// Transposes the rows and columns of a matrix
template<typename T>
static void transpose_regular(T* dest, const T *src, int rows, int cols) {
    int prod = rows*cols;

    #pragma omp parallel for
    for(int m = 0; m < prod; ++m) {
        int i = m / rows;
        int j = m % rows;
        dest[m] = src[i + cols*j];
    }
}

inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0*lda]);
    __m128 row2 = _mm_load_ps(&A[1*lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0*ldb], row1);
     _mm_store_ps(&B[1*ldb], row2);
     _mm_store_ps(&B[2*ldb], row3);
     _mm_store_ps(&B[3*ldb], row4);
}

// Intrinsics + parallel
inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}

void fixed_buffer_map::transpose(int batch_size)
{
    if(batch_size <= 0)
    {
        throw invalid_argument("batch_size: batch size must be greater than 0");
    }

    for (auto name : m_names)
    {
        //char* src = this->operator[](name)->data();
        buffer_fixed_size_elements* bfse = this->operator[](name);
        char* src = bfse->data();
        int size = this->operator[](name)->size();
        int cols = size / batch_size;
        char* dest = new char [size];

        int element_size = (this->operator[](name))->get_shape_type().get_otype().get_size();
        switch(element_size)
        {
            case 1:
            {
                transpose_regular<uint8_t>((uint8_t*)dest, (uint8_t*)src, batch_size, cols);
//                int lda = ROUND_UP(cols, 16);
//                int ldb = ROUND_UP(batch_size, 16);
//                int block_size = 16;
//                cout << batch_size << " " << cols << " " << block_size << endl;
//                transpose_block_SSE4x4((float*)dest, (float*)src, batch_size, cols, lda, ldb, block_size);
                break;
            }
            case 2:
                transpose_regular<uint16_t>((uint16_t*)dest, (uint16_t*)src, batch_size, cols/2);
                break;
            case 4:
            {
                transpose_regular<uint32_t>((uint32_t*)dest, (uint32_t*)src, batch_size, cols/4);
                break;
            }
            case 8:
                transpose_regular<uint64_t>((uint64_t*)dest, (uint64_t*)src, batch_size, cols/8);
                break;
            default:
                throw "unsupported type";
        }
        
        //transpose_block_SSE4x4((float*)buf->data(), (float*)buf2, batch_size, cols, lda, ldb, block_size);
        
        //memcpy(src, dest, size);
        bfse->swap(dest);
        
        //delete[] dest;
    }
}

void encoded_record_list::transpose(int batch_size)
{
    if(batch_size <= 0)
    {
        throw invalid_argument("batch_size: batch size must be greater than 0");
    }
    
    for(int i=0; i<m_records.size(); ++i)
    {
        //variable_record_field& rc = m_records[i].size(i);
        //cout << rc.size() << endl;
    }
}
