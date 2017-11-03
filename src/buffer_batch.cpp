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
#include <immintrin.h>

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
enum TransposeType { REGULAR, SSE, AVX2 };

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

// Transposes the rows and columns of a matrix (Adrian opitmized)
template<typename T>
static void transpose_regular2(T* dest, const T *src, int rows, int cols)
{
    //#pragma omp parallel for
    int dst_indx = 0;
    int src_indx = 0;
    for(int c = 0; c < cols; ++c)
    {
        src_indx = c;
        for(int r = 0; r < rows; ++r)
        {
            dest[dst_indx] = src[src_indx];
            dst_indx++;
            src_indx += cols;
        }
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

// SSE 128
inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
    //#pragma omp parallel for
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

// AVX2 256
/*
inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

inline void transpose8x8_AVX2(float *A, float *B, const int lda, const int ldb) {
    __m256 row0 = _mm256_load_ps(&A[0*lda]);
    __m256 row1 = _mm256_load_ps(&A[1*lda]);
    __m256 row2 = _mm256_load_ps(&A[2*lda]);
    __m256 row3 = _mm256_load_ps(&A[3*lda]);
    __m256 row4 = _mm256_load_ps(&A[4*lda]);
    __m256 row5 = _mm256_load_ps(&A[5*lda]);
    __m256 row6 = _mm256_load_ps(&A[6*lda]);
    __m256 row7 = _mm256_load_ps(&A[7*lda]);    
    transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
    _mm256_store_ps(&B[0*ldb], row0);
    _mm256_store_ps(&B[1*ldb], row1);
    _mm256_store_ps(&B[2*ldb], row2);
    _mm256_store_ps(&B[3*ldb], row3);
    _mm256_store_ps(&B[4*ldb], row4);
    _mm256_store_ps(&B[5*ldb], row5);
    _mm256_store_ps(&B[6*ldb], row6);
    _mm256_store_ps(&B[7*ldb], row7);
}

inline void transpose_block_AVX2(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
    //#pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose8x8_AVX2(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}
*/

static void transpose_buf(char* dest, char* src, size_t rows, size_t cols, size_t element_size, TransposeType type)
{
    int lda = ROUND_UP(cols, 16);
    int ldb = ROUND_UP(rows, 16);
    int block_size = 64;
    
    switch(element_size)
    {
        case 1:
        {
            if(type == TransposeType::REGULAR) transpose_regular<uint8_t>(reinterpret_cast<uint8_t*>(dest), reinterpret_cast<uint8_t*>(src), rows, cols);
            else if(type == TransposeType::SSE) transpose_block_SSE4x4(reinterpret_cast<float*>(src), reinterpret_cast<float*>(dest), rows/4, cols/4, lda, ldb, block_size);
            //else if(type == TransposeType::AVX2) transpose_block_AVX2(reinterpret_cast<float*>(src), reinterpret_cast<float*>(dest), rows/4, cols/4, lda, ldb, block_size);
            break;
        }
        case 2:
            transpose_regular<uint16_t>(reinterpret_cast<uint16_t*>(dest), reinterpret_cast<uint16_t*>(src), rows, cols/2);
            break;
        case 4:
        {
            transpose_regular<uint32_t>(reinterpret_cast<uint32_t*>(dest), reinterpret_cast<uint32_t*>(src), rows, cols/4);
            break;
        }
        case 8:
            transpose_regular<uint64_t>(reinterpret_cast<uint64_t*>(dest), reinterpret_cast<uint64_t*>(src), rows, cols/8);
            break;
        default:
            throw "unsupported type";
    }
}

void fixed_buffer_map::transpose(size_t batch_size)
{
    if(batch_size <= 0)
    {
        throw invalid_argument("batch_size: batch size must be greater than 0");
    }

    for (auto name : m_names)
    {
        buffer_fixed_size_elements* bfse = this->operator[](name);
        char* src = bfse->data();
        int size = this->operator[](name)->size();
        int cols = size / batch_size;
        char* dest = new char [size];

        int element_size = (this->operator[](name))->get_shape_type().get_otype().get_size();
        
        transpose_buf(dest, src, batch_size, cols, element_size, TransposeType::SSE);

        bfse->swap(dest);
    }
}

void fixed_buffer_map::copy_with_transpose(fixed_buffer_map& src, size_t src_index, size_t dst_index, size_t count, size_t batch_size)
{
    if(batch_size <= 0)
    {
        throw invalid_argument("batch_size: batch size must be greater than 0");
    }

    for (auto name : m_names)
    {
        buffer_fixed_size_elements* src_fbm = src[name];
        buffer_fixed_size_elements* dst_fbm = operator[](name);
        char*                       p_src   = src_fbm->get_item(src_index);
        char*                       p_dst   = dst_fbm->get_item(dst_index);

        if ((count + src_index > src_fbm->get_item_count()) ||
            (count + dst_index > dst_fbm->get_item_count()))
            throw invalid_argument("buffer_fixed_size: count out-of-range");

        int element_size = (this->operator[](name))->get_shape_type().get_otype().get_size();
        
        transpose_buf(p_dst, p_src, batch_size, count * src_fbm->get_stride() / batch_size, element_size, TransposeType::SSE);
    }
}
