#include <stdint.h>
#include <immintrin.h>

namespace transpose {
    namespace sse {

        #define combine_4_2bits(n0, n1, n2, n3) (n0 + (n1<<2) + (n2<<4) + (n3<<6))
        #define _128_shuffle(x, y, n0, n1, n2, n3) _mm_shuffle_ps(x, y, combine_4_2bits (n0, n1, n2, n3))
        #define _128i_shuffle(x, y, n0, n1, n2, n3) _mm_castps_si128(_128_shuffle(_mm_castsi128_ps(x), _mm_castsi128_ps(y), n0, n1, n2, n3))

        inline void _128i_store (unsigned char * p, __m128i x)
        {
            _mm_store_si128 ((__m128i *) p, x);
        }

        inline __m128i _128i_load (const unsigned char * p)
        {
            return _mm_load_si128 ((const __m128i *) p);
        }

        inline __m128i transpose_4x4 (__m128i m)
        {
            __m128i y = _mm_setr_epi8 (0, 4, 8, 12,
                                       1, 5, 9, 13,
                                       2, 6, 10, 14,
                                       3, 7, 11, 15);

            return _mm_shuffle_epi8(m, y);
        }

        inline void transpose_4x4_dwords (__m128i w0, __m128i w1,
                                          __m128i w2, __m128i w3,
                                          __m128i &r0, __m128i &r1,
                                          __m128i &r2, __m128i &r3)
        {
            // 0  1  2  3
            // 4  5  6  7
            // 8  9  10 11
            // 12 13 14 15
            __m128i x0 = _128i_shuffle (w0, w1, 0, 1, 0, 1); // 0 1 4 5
            __m128i x1 = _128i_shuffle (w0, w1, 2, 3, 2, 3); // 2 3 6 7
            __m128i x2 = _128i_shuffle (w2, w3, 0, 1, 0, 1); // 8 9 12 13
            __m128i x3 = _128i_shuffle (w2, w3, 2, 3, 2, 3); // 10 11 14 15
            r0 = _128i_shuffle (x0, x2, 0, 2, 0, 2);
            r1 = _128i_shuffle (x0, x2, 1, 3, 1, 3);
            r2 = _128i_shuffle (x1, x3, 0, 2, 0, 2);
            r3 = _128i_shuffle (x1, x3, 1, 3, 1, 3);
        }

        inline void transpose_16x16 (
                        __m128i&  x0, __m128i&  x1, __m128i&  x2, __m128i&  x3,
                        __m128i&  x4, __m128i&  x5, __m128i&  x6, __m128i&  x7,
                        __m128i&  x8, __m128i&  x9, __m128i& x10, __m128i& x11,
                        __m128i& x12, __m128i& x13, __m128i& x14, __m128i& x15)
        {
            __m128i w00, w01, w02, w03;
            __m128i w10, w11, w12, w13;
            __m128i w20, w21, w22, w23;
            __m128i w30, w31, w32, w33;
            transpose_4x4_dwords ( x0,  x1,  x2,  x3, w00, w01, w02, w03);
            transpose_4x4_dwords ( x4,  x5,  x6,  x7, w10, w11, w12, w13);
            transpose_4x4_dwords ( x8,  x9, x10, x11, w20, w21, w22, w23);
            transpose_4x4_dwords (x12, x13, x14, x15, w30, w31, w32, w33);
            w00 = transpose_4x4 (w00);
            w01 = transpose_4x4 (w01);
            w02 = transpose_4x4 (w02);
            w03 = transpose_4x4 (w03);
            w10 = transpose_4x4 (w10);
            w11 = transpose_4x4 (w11);
            w12 = transpose_4x4 (w12);
            w13 = transpose_4x4 (w13);
            w20 = transpose_4x4 (w20);
            w21 = transpose_4x4 (w21);
            w22 = transpose_4x4 (w22);
            w23 = transpose_4x4 (w23);
            w30 = transpose_4x4 (w30);
            w31 = transpose_4x4 (w31);
            w32 = transpose_4x4 (w32);
            w33 = transpose_4x4 (w33);
            transpose_4x4_dwords (w00, w10, w20, w30,  x0,  x1,  x2, x3);
            transpose_4x4_dwords (w01, w11, w21, w31,  x4,  x5,  x6, x7);
            transpose_4x4_dwords (w02, w12, w22, w32,  x8,  x9, x10, x11);
            transpose_4x4_dwords (w03, w13, w23, w33, x12, x13, x14, x15);
        }

        void transpose(uint8_t* dest, const uint8_t *src, int rows, int cols)
        {
            const int block_size = 16;
            const int rows_block = rows / block_size;
            const int cols_block = cols / block_size;

            for(int cb = 0; cb < cols; cb += block_size)
            {
                for(int rb = 0; rb < rows; rb += block_size)
                {
                    const uint8_t *src_c = src + rb*cols + cb;
                    uint8_t *dst_c = dest + cb*rows + rb;

                    __m128i row0 = _128i_load(src_c); src_c += cols;
                    __m128i row1 = _128i_load(src_c); src_c += cols;
                    __m128i row2 = _128i_load(src_c); src_c += cols;
                    __m128i row3 = _128i_load(src_c); src_c += cols;
                    __m128i row4 = _128i_load(src_c); src_c += cols;
                    __m128i row5 = _128i_load(src_c); src_c += cols;
                    __m128i row6 = _128i_load(src_c); src_c += cols;
                    __m128i row7 = _128i_load(src_c); src_c += cols;
                    __m128i row8 = _128i_load(src_c); src_c += cols;
                    __m128i row9 = _128i_load(src_c); src_c += cols;
                    __m128i row10 = _128i_load(src_c); src_c += cols;
                    __m128i row11 = _128i_load(src_c); src_c += cols;
                    __m128i row12 = _128i_load(src_c); src_c += cols;
                    __m128i row13 = _128i_load(src_c); src_c += cols;
                    __m128i row14 = _128i_load(src_c); src_c += cols;
                    __m128i row15 = _128i_load(src_c); src_c += cols;

                    transpose_16x16(row0,row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12,row13,row14,row15);

                     _128i_store(dst_c, row0); dst_c += rows;
                     _128i_store(dst_c, row1); dst_c += rows;
                     _128i_store(dst_c, row2); dst_c += rows;
                     _128i_store(dst_c, row3); dst_c += rows;
                     _128i_store(dst_c, row4); dst_c += rows;
                     _128i_store(dst_c, row5); dst_c += rows;
                     _128i_store(dst_c, row6); dst_c += rows;
                     _128i_store(dst_c, row7); dst_c += rows;
                     _128i_store(dst_c, row8); dst_c += rows;
                     _128i_store(dst_c, row9); dst_c += rows;
                     _128i_store(dst_c, row10); dst_c += rows;
                     _128i_store(dst_c, row11); dst_c += rows;
                     _128i_store(dst_c, row12); dst_c += rows;
                     _128i_store(dst_c, row13); dst_c += rows;
                     _128i_store(dst_c, row14); dst_c += rows;
                     _128i_store(dst_c, row15); dst_c += rows;
                }
            }
        }
    }}
