#ifndef USHITORA_ANQOU_AQTFHE2_AQTFHE2_HPP
#define USHITORA_ANQOU_AQTFHE2_AQTFHE2_HPP

#include <array>
#include <cassert>
#include <memory>
#include <random>

namespace aqtfhe2 {
namespace params {
struct CGGI16 {
    static constexpr uint32_t n()
    {
        return 500;
    }
    static constexpr double alpha()
    {
        return 2.44e-5;
    }
    static constexpr uint32_t Nbit()
    {
        return 10;
    }
    static constexpr uint32_t N()
    {
        return 1u << Nbit();
    }
    static constexpr uint32_t l()
    {
        return 2;
    }
    static constexpr uint32_t Bgbit()
    {
        return 10;
    }
    static constexpr uint32_t Bg()
    {
        return 1u << Bgbit();
    }
    static constexpr double alphabk()
    {
        return 3.73e-9;
    }
    static constexpr uint32_t t()
    {
        return 8;
    }
    static constexpr uint32_t basebit()
    {
        return 2;
    }
    static constexpr double alphaks()
    {
        return 2.44e-5;
    }
    static constexpr uint32_t mu()
    {
        return 1U << 29;
    }
};

struct CGGI19 {
    static constexpr uint32_t n()
    {
        return 630;
    }
    static constexpr double alpha()
    {
        return 3.0517578125e-05;  // 2^(-15)
    }
    static constexpr uint32_t Nbit()
    {
        return 10;
    }
    static constexpr uint32_t N()
    {
        return 1 << Nbit();
    }
    static constexpr uint32_t l()
    {
        return 3;
    }
    static constexpr uint32_t Bgbit()
    {
        return 6;
    }
    static constexpr uint32_t Bg()
    {
        return 1 << Bgbit();
    }
    static constexpr double alphabk()
    {
        return 2.98023223876953125e-08;  // 2^(-25)
    }
    static constexpr uint32_t t()
    {
        return 8;
    }
    static constexpr uint32_t basebit()
    {
        return 2;
    }
    static constexpr double alphaks()
    {
        return alpha();
    }
    static constexpr uint32_t mu()
    {
        return 1U << 29;
    }
};
}  // namespace params

namespace detail {

template <class T, size_t Size>
class array_slice {
public:
    using iterator = typename std::array<T, Size>::iterator;
    using const_iterator = typename std::array<T, Size>::const_iterator;
    using reference = typename std::array<T, Size>::reference;
    using const_reference = typename std::array<T, Size>::const_reference;
    using pointer = typename std::array<T, Size>::pointer;
    using const_pointer = typename std::array<T, Size>::const_pointer;

private:
    iterator head_;

public:
    array_slice(iterator head) : head_(std::move(head))
    {
    }

    static constexpr size_t size() noexcept
    {
        return Size;
    }

    constexpr pointer data() noexcept
    {
        return &*head_;
    }
    constexpr const_pointer data() const noexcept
    {
        return &*head_;
    }

    constexpr reference operator[](size_t i)
    {
        return *(head_ + i);
    }
    constexpr const_reference operator[](size_t i) const
    {
        return *(head_ + i);
    }

    constexpr iterator begin() noexcept
    {
        return head_;
    }
    constexpr const_iterator begin() const noexcept
    {
        return head_;
    }

    constexpr iterator end() noexcept
    {
        return head_ + Size;
    }
    constexpr const_iterator end() const noexcept
    {
        return head_ + Size;
    }
};

template <class T, size_t Size>
class const_array_slice {
public:
    using const_iterator = typename std::array<T, Size>::const_iterator;
    using const_reference = typename std::array<T, Size>::const_reference;
    using const_pointer = typename std::array<T, Size>::const_pointer;

private:
    const_iterator head_;

public:
    const_array_slice(const_iterator head) : head_(head)
    {
    }
    const_array_slice(array_slice<T, Size> nonconst) : head_(nonconst.begin())
    {
    }

    static constexpr size_t size() noexcept
    {
        return Size;
    }

    constexpr const_pointer data() const noexcept
    {
        return &*head_;
    }

    constexpr const_reference operator[](size_t i) const
    {
        return *(head_ + i);
    }

    constexpr const_iterator begin() const noexcept
    {
        return head_;
    }

    constexpr const_iterator end() const noexcept
    {
        return head_ + Size;
    }
};

template <class T, size_t... Shape>
class nd_array {
    static_assert(sizeof...(Shape) >= 1, "Empty nd_array is invalid.");

public:
    static constexpr size_t size() noexcept
    {
        return (Shape * ...);
    }
    static constexpr std::array<size_t, sizeof...(Shape)> shape() noexcept
    {
        return {Shape...};
    }
    static constexpr size_t dim() noexcept
    {
        return sizeof...(Shape);
    }

public:
    using container = std::array<T, size()>;
    using reference = typename container::reference;
    using const_reference = typename container::const_reference;
    using iterator = typename container::iterator;
    using const_iterator = typename container::const_iterator;

    container data_;

private:
    template <size_t N, class... Indices>
    constexpr size_t index_impl(size_t sum, size_t head, Indices... tail) const
        noexcept
    {
        static_assert(N < dim());
        return index_impl<N + 1>(head + sum * shape()[N], tail...);
    }
    template <size_t N>
    constexpr size_t index_impl(size_t sum) const noexcept
    {
        static_assert(N == dim());
        return sum;
    }
    template <class... Indices>
    constexpr size_t index(size_t head, Indices... tail) const noexcept
    {
        return index_impl<0>(0, head, tail...);
    }

public:
    template <class... Indices>
    constexpr reference operator()(Indices... indices)
    {
        assert(index(indices...) < size());
        return data_[index(indices...)];
    }
    template <class... Indices>
    constexpr const_reference operator()(Indices... indices) const
    {
        assert(index(indices...) < size());
        return data_[index(indices...)];
    }
    template <class Index, std::enable_if_t<dim() == 1, Index> = 0>
    constexpr reference operator[](Index i)
    {
        return data_[i];
    }
    template <class Index, std::enable_if_t<dim() == 1, Index> = 0>
    constexpr const_reference operator[](Index i) const
    {
        return data_[i];
    }

    constexpr iterator begin() noexcept
    {
        return data_.begin();
    }
    constexpr const_iterator begin() const noexcept
    {
        return data_.begin();
    }
    constexpr const_iterator cbegin() const noexcept
    {
        return data_.cbegin();
    }

    constexpr iterator end() noexcept
    {
        return data_.end();
    }
    constexpr const_iterator end() const noexcept
    {
        return data_.end();
    }
    constexpr const_iterator cend() const noexcept
    {
        return data_.cend();
    }

    constexpr T *data() noexcept
    {
        return data_.data();
    }
    constexpr const T *data() const noexcept
    {
        return data_.data();
    }

    template <class... Indices>
    constexpr array_slice<T, shape()[dim() - 1]> slice(
        Indices... indices) noexcept
    {
        return array_slice<T, shape()[dim() - 1]>{begin() +
                                                  this->index(indices..., 0)};
    }
    template <class... Indices>
    constexpr const_array_slice<T, shape()[dim() - 1]> slice(
        Indices... indices) const noexcept
    {
        return const_array_slice<T, shape()[dim() - 1]>{
            begin() + this->index(indices..., 0)};
    }
    template <class... Indices>
    constexpr const_array_slice<T, shape()[dim() - 1]> cslice(
        Indices... indices) const noexcept
    {
        return const_array_slice<T, shape()[dim() - 1]>{
            begin() + this->index(indices..., 0)};
    }

    bool operator==(const nd_array<T, Shape...> &rhs)
    {
        return data_ == rhs.data_;
    }
    bool operator!=(const nd_array<T, Shape...> &rhs)
    {
        return !(data_ == rhs.data_);
    }
};

inline uint32_t double2torus32(double d)
{
    return (d - std::floor(d)) * std::pow(2, 32);
}

template <size_t N>
inline nd_array<double, N> mul_in_fd(const_array_slice<double, N> a,
                                     const_array_slice<double, N> b)
{
    nd_array<double, N> ret;
    for (size_t i = 0; i < N / 2; i++) {
        double aimbim = a[i + N / 2] * b[i + N / 2];
        double arebim = a[i] * b[i + N / 2];
        ret[i] = a[i] * b[i] - aimbim;
        ret[i + N / 2] = a[i + N / 2] * b[i] + arebim;
    }
    return ret;
}

template <size_t N>
inline void fma_in_fd(array_slice<double, N> res,
                      const_array_slice<double, N> a,
                      const_array_slice<double, N> b)
{
    for (size_t i = 0; i < N / 2; i++) {
        res[i] = a[i + N / 2] * b[i + N / 2] - res[i];
        res[i] = a[i] * b[i] - res[i];
        res[i + N / 2] += a[i] * b[i + N / 2];
        res[i + N / 2] += a[i + N / 2] * b[i];
    }
}

inline uint32_t mod_switch_from_torus32(uint32_t Msize, uint32_t phase)
{
    uint64_t interv = ((1UL << 63) / Msize) * 2;  // width of each intervall
    uint64_t half_interval = interv / 2;  // begin of the first intervall
    uint64_t phase64 = (static_cast<uint64_t>(phase) << 32) + half_interval;
    // floor to the nearest multiples of interv
    return static_cast<uint32_t>(phase64 / interv);
}

/*
    NOTE: The following FFT implementation (from "FROM HERE" to "TO HERE") is
    copied and modified from TFHE's spqlios.
    See TFHE's source code for the details (https://github.com/tfhe/tfhe).
    TFHE is licensed under Apache-2.0:

        Copyright 2016 - Nicolas Gama <nicolas.gama@gmail.com> et al.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

    ----- FFT IMPLEMENTATION FROM HERE -----
*/
inline double accurate_cos(int32_t i, int32_t n)
{
    i = ((i % n) + n) % n;
    const double nd = n;
    const double pi = 3.1415926535897932384626433832795028;

    if (i >= 3 * n / 4)
        return std::cos(2. * pi * (n - i) / nd);
    if (i >= 2 * n / 4)
        return -std::cos(2. * pi * (i - n / 2) / nd);
    if (i >= 1 * n / 4)
        return -std::cos(2. * pi * (n / 2 - i) / nd);
    return std::cos(2. * pi * (i) / nd);
}

inline double accurate_sin(int32_t i, int32_t n)
{
    i = ((i % n) + n) % n;
    const double nd = n;
    const double pi = 3.1415926535897932384626433832795028;

    if (i >= 3 * n / 4)
        return -sin(2. * pi * (n - i) / nd);
    if (i >= 2 * n / 4)
        return -sin(2. * pi * (i - n / 2) / nd);
    if (i >= 1 * n / 4)
        return sin(2. * pi * (n / 2 - i) / nd);
    return sin(2. * pi * (i) / nd);
}

template <class RandomAccessIterator0, class RandomAccessIterator1,
          class RandomAccessIterator2>
inline void dotp4(RandomAccessIterator0 res, RandomAccessIterator1 a,
                  RandomAccessIterator2 b)
{
    for (size_t i = 0; i < 4; i++)
        res[i] = a[i] * b[i];
}

template <class RandomAccessIterator0, class RandomAccessIterator1,
          class RandomAccessIterator2>
inline void add4(RandomAccessIterator0 res, RandomAccessIterator1 a,
                 RandomAccessIterator2 b)
{
    for (size_t i = 0; i < 4; i++)
        res[i] = a[i] + b[i];
}

template <class RandomAccessIterator0, class RandomAccessIterator1,
          class RandomAccessIterator2>
inline void sub4(RandomAccessIterator0 res, RandomAccessIterator1 a,
                 RandomAccessIterator2 b)
{
    for (size_t i = 0; i < 4; i++)
        res[i] = a[i] - b[i];
}

template <class RandomAccessIterator0, class RandomAccessIterator1>
inline void copy4(RandomAccessIterator0 res, RandomAccessIterator1 a)
{
    for (size_t i = 0; i < 4; i++)
        res[i] = a[i];
}

template <size_t N>
class fft_processor {
    static_assert(N >= 16, "N must be >=16");
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");

private:
    nd_array<double, 2 * N> tables_direct_, tables_reverse_;

private:
    void fft(array_slice<double, N> out)
    {
        constexpr size_t n = N * 2, ns4 = n / 4;
        double tmp0[4], tmp1[4], tmp2[4], tmp3[4];
        auto itre = out.begin(), itim = out.begin() + ns4;

        // size 2
        for (size_t block = 0; block < ns4; block += 4) {
            auto re = itre + block;
            auto im = itim + block;

            tmp0[0] = re[0] + re[1];
            tmp0[1] = re[0] - re[1];
            tmp0[2] = re[2] + re[3];
            tmp0[3] = re[2] - re[3];
            copy4(re, tmp0);

            tmp1[0] = im[0] + im[1];
            tmp1[1] = im[0] - im[1];
            tmp1[2] = im[2] + im[3];
            tmp1[3] = im[2] - im[3];
            copy4(im, tmp1);
        }

        // size 4
        for (size_t block = 0; block < ns4; block += 4) {
            auto re = itre + block;
            auto im = itim + block;

            tmp0[0] = re[0] + re[2];
            tmp0[1] = re[1] + im[3];
            tmp0[2] = re[0] - re[2];
            tmp0[3] = re[1] - im[3];

            tmp1[0] = im[0] + im[2];
            tmp1[1] = im[1] - re[3];
            tmp1[2] = im[0] - im[2];
            tmp1[3] = im[1] + re[3];

            copy4(re, tmp0);
            copy4(im, tmp1);
        }

        // general loop
        auto cur_tt = tables_direct_.cbegin();
        for (size_t halfnn = 4; halfnn < ns4; halfnn *= 2) {
            size_t nn = 2 * halfnn;
            for (size_t block = 0; block < ns4; block += nn) {
                for (size_t off = 0; off < halfnn; off += 4) {
                    auto re0 = itre + block + off;
                    auto im0 = itim + block + off;
                    auto re1 = itre + block + halfnn + off;
                    auto im1 = itim + block + halfnn + off;
                    auto tcs = cur_tt + 2 * off;
                    auto tsn = tcs + 4;

                    dotp4(tmp0, re1, tcs);   // re*cos
                    dotp4(tmp1, re1, tsn);   // re*sin
                    dotp4(tmp2, im1, tcs);   // im*cos
                    dotp4(tmp3, im1, tsn);   // im*sin
                    sub4(tmp0, tmp0, tmp3);  // re2
                    add4(tmp1, tmp1, tmp2);  // im2
                    add4(tmp2, re0, tmp0);   // re + re
                    add4(tmp3, im0, tmp1);   // im + im
                    sub4(tmp0, re0, tmp0);   // re - re
                    sub4(tmp1, im0, tmp1);   // im - im
                    copy4(re0, tmp2);
                    copy4(im0, tmp3);
                    copy4(re1, tmp0);
                    copy4(im1, tmp1);
                }
            }
            cur_tt += nn;
        }

        // multiply by omb^j
        for (size_t j = 0; j < ns4; j += 4) {
            auto r0 = cur_tt + 2 * j;
            auto r1 = r0 + 4;
            //(re*cos-im*sin) + i (im*cos+re*sin)
            auto d0 = itre + j;
            auto d1 = itim + j;

            dotp4(tmp0, d0, r0);  // re*cos
            dotp4(tmp1, d1, r0);  // im*cos
            dotp4(tmp2, d0, r1);  // re*sin
            dotp4(tmp3, d1, r1);  // im*sin
            sub4(d0, tmp0, tmp3);
            add4(d1, tmp1, tmp2);
        }
    }

    void ifft(array_slice<double, N> out)
    {
        constexpr size_t n = N * 2, ns4 = n / 4;
        double tmp0[4], tmp1[4], tmp2[4], tmp3[4];
        auto itre = out.begin(), itim = out.begin() + ns4;

        // multiply by omb^j
        for (size_t j = 0; j < ns4; j += 4) {
            auto r0 = tables_reverse_.cbegin() + 2 * j;
            auto r1 = r0 + 4;
            //(re*cos-im*sin) + i (im*cos+re*sin)
            auto d0 = itre + j;
            auto d1 = itim + j;

            dotp4(tmp0, d0, r0);  // re*cos
            dotp4(tmp1, d1, r0);  // im*cos
            dotp4(tmp2, d0, r1);  // re*sin
            dotp4(tmp3, d1, r1);  // im*sin
            sub4(d0, tmp0, tmp3);
            add4(d1, tmp1, tmp2);
        }

        // at the beginning of iteration nn
        // a_{j,i} has P_{i%nn}(omega^j)
        // where j between [rev(1) and rev(3)[
        // and i between [0 and nn[
        auto cur_tt = tables_reverse_.cbegin();
        for (size_t nn = ns4; nn >= 8; nn /= 2) {
            size_t halfnn = nn / 2;
            cur_tt += 2 * nn;
            for (size_t block = 0; block < ns4; block += nn) {
                for (size_t off = 0; off < halfnn; off += 4) {
                    auto d00 = itre + block + off;
                    auto d01 = itim + block + off;
                    auto d10 = itre + block + halfnn + off;
                    auto d11 = itim + block + halfnn + off;
                    add4(tmp0, d00, d10);  // re + re
                    add4(tmp1, d01, d11);  // im + im
                    sub4(tmp2, d00, d10);  // re - re
                    sub4(tmp3, d01, d11);  // im - im
                    copy4(d00, tmp0);
                    copy4(d01, tmp1);
                    auto r0 = cur_tt + 2 * off;
                    auto r1 = r0 + 4;
                    dotp4(tmp0, tmp2, r0);  // re*cos
                    dotp4(tmp1, tmp3, r1);  // im*sin
                    sub4(d10, tmp0, tmp1);
                    dotp4(tmp0, tmp2, r1);  // re*sin
                    dotp4(tmp1, tmp3, r0);  // im*cos
                    add4(d11, tmp0, tmp1);
                }
            }
        }

        // size 4
        for (size_t block = 0; block < ns4; block += 4) {
            auto re = itre + block;
            auto im = itim + block;

            tmp0[0] = re[0] + re[2];
            tmp0[1] = re[1] + re[3];
            tmp0[2] = re[0] - re[2];
            tmp0[3] = im[3] - im[1];

            tmp1[0] = im[0] + im[2];
            tmp1[1] = im[1] + im[3];
            tmp1[2] = im[0] + -im[2];
            tmp1[3] = re[1] + -re[3];

            copy4(re, tmp0);
            copy4(im, tmp1);
        }

        // size 2
        for (size_t block = 0; block < ns4; block += 4) {
            auto re = itre + block;
            auto im = itim + block;

            tmp0[0] = re[0] + re[1];
            tmp0[1] = re[0] - re[1];
            tmp0[2] = re[2] + re[3];
            tmp0[3] = re[2] - re[3];
            copy4(re, tmp0);

            tmp1[0] = im[0] + im[1];
            tmp1[1] = im[0] - im[1];
            tmp1[2] = im[2] + im[3];
            tmp1[3] = im[2] - im[3];
            copy4(im, tmp1);
        }
    }

public:
    fft_processor()
    {
        // new fft table
        {
            constexpr size_t n = 2 * N, ns4 = n / 4;
            auto it = tables_direct_.begin();
            for (size_t halfnn = 4; halfnn < ns4; halfnn *= 2) {
                const size_t nn = 2 * halfnn;
                const size_t j = n / nn;
                for (size_t i = 0; i < halfnn; i += 4) {
                    for (size_t k = 0; k < 4; k++)
                        *it++ = accurate_cos(-j * (i + k), n);
                    for (size_t k = 0; k < 4; k++)
                        *it++ = accurate_sin(-j * (i + k), n);
                }
            }
            // last iteration
            for (size_t i = 0; i < ns4; i += 4) {
                for (size_t k = 0; k < 4; k++)
                    *it++ = accurate_cos(-(i + k), n);
                for (size_t k = 0; k < 4; k++)
                    *it++ = accurate_sin(-(i + k), n);
            }
        }

        // new ifft table
        {
            constexpr size_t n = 2 * N, ns4 = n / 4;
            auto it = tables_reverse_.begin();
            // first iteration
            for (size_t j = 0; j < ns4; j += 4) {
                for (size_t k = 0; k < 4; k++)
                    *it++ = accurate_cos(j + k, n);
                for (size_t k = 0; k < 4; k++)
                    *it++ = accurate_sin(j + k, n);
            }
            // subsequent iterations
            for (size_t nn = ns4; nn >= 8; nn /= 2) {
                size_t halfnn = nn / 2;
                size_t j = n / nn;
                for (size_t i = 0; i < halfnn; i += 4) {
                    for (size_t k = 0; k < 4; k++)
                        *it++ = accurate_cos(j * (i + k), n);
                    for (size_t k = 0; k < 4; k++)
                        *it++ = accurate_sin(j * (i + k), n);
                }
            }
        }
    }

    void twist_fft_lvl1(array_slice<uint32_t, N> out,
                        const_array_slice<double, N> a)
    {
        nd_array<double, N> tmp;
        for (size_t i = 0; i < N; i++)
            tmp[i] = a[i] * (2.0 / N);
        fft(tmp.slice());
        for (size_t i = 0; i < N; i++)
            out[i] = static_cast<uint32_t>(static_cast<int64_t>(tmp[i]));
    }
    void twist_ifft_lvl1(array_slice<double, N> out,
                         const_array_slice<uint32_t, N> a)
    {
        nd_array<double, N> tmp;
        for (size_t i = 0; i < N; i++)
            tmp[i] = static_cast<int32_t>(a[i]);
        ifft(tmp.slice());
        for (size_t i = 0; i < N; i++)
            out[i] = tmp[i];
    }

    nd_array<uint32_t, N> twist_fft_lvl1(const nd_array<double, N> &a)
    {
        nd_array<uint32_t, N> ret;
        twist_fft_lvl1(ret.slice(), a.slice());
        return ret;
    }
    nd_array<double, N> twist_ifft_lvl1(const nd_array<uint32_t, N> &a)
    {
        nd_array<double, N> ret;
        twist_ifft_lvl1(ret.slice(), a.slice());
        return ret;
    }
};
/*
   ----- FFT IMPLEMENTATION TO HERE -----
*/

}  // namespace detail

// Forward declarations
template <class P>
class tlwe_lvl0;
template <class P>
class decomposed_trlwe_in_fd_lvl1;
template <class P>
class trgswfft_lvl1;

template <class P>
class secret_key {
public:
    using container_key_lvl0 = detail::nd_array<uint32_t, P::n()>;
    using container_key_lvl1 = detail::nd_array<uint32_t, P::N()>;

private:
    container_key_lvl0 key_lvl0_;
    container_key_lvl1 key_lvl1_;

public:
    explicit secret_key(container_key_lvl0 key_lvl0,
                        container_key_lvl1 key_lvl1)
        : key_lvl0_(std::move(key_lvl0)), key_lvl1_(std::move(key_lvl1))
    {
    }

    template <class RandomEngine>
    static secret_key make(RandomEngine &rand)
    {
        std::binomial_distribution<uint32_t> binary;

        container_key_lvl0 key_lvl0;
        for (uint32_t &v : key_lvl0)
            v = binary(rand);

        container_key_lvl1 key_lvl1;
        for (uint32_t &v : key_lvl1)
            v = binary(rand);

        return secret_key{std::move(key_lvl0), std::move(key_lvl1)};
    }

    template <class RandomEngine>
    tlwe_lvl0<P> encrypt(RandomEngine &rand, bool plain) const
    {
        return tlwe_lvl0<P>::boots_sym_encrypt(rand, key_lvl0_, plain);
    }

    bool decrypt(const tlwe_lvl0<P> &src) const
    {
        return src.boots_sym_decrypt(key_lvl0_);
    }

    const container_key_lvl0 &key_lvl0() const noexcept
    {
        return key_lvl0_;
    }
    const container_key_lvl1 &key_lvl1() const noexcept
    {
        return key_lvl1_;
    }
};

template <class P>
class key_switching_key {
public:
    using container = detail::nd_array<uint32_t, P::N(), P::t(),
                                       (1 << P::basebit()) - 1, P::n() + 1>;

private:
    container data_;

public:
    template <class RandomEngine>
    explicit key_switching_key(
        RandomEngine &rand,
        const typename secret_key<P>::container_key_lvl0 &key_lvl0,
        const typename secret_key<P>::container_key_lvl1 &key_lvl1);

    uint32_t operator()(size_t i, size_t j, size_t k, size_t l) const
    {
        return data_(i, j, k, l);
    }
};

template <class P>
class cloud_key {
private:
    std::unique_ptr<key_switching_key<P>> ksk_;
    std::vector<trgswfft_lvl1<P>> bkfftlvl01_;

public:
    template <class RandomEngine>
    cloud_key(RandomEngine &rand,
              const typename secret_key<P>::container_key_lvl0 &key_lvl0,
              const typename secret_key<P>::container_key_lvl1 &key_lvl1)
        : ksk_(std::make_unique<key_switching_key<P>>(rand, key_lvl0, key_lvl1))
    {
        // Fill bkfftlvl01_
        bkfftlvl01_.reserve(P::n());
        for (size_t i = 0; i < P::n(); i++) {
            bkfftlvl01_.push_back(trgswfft_lvl1<P>::sym_encrypt(
                rand, key_lvl1, P::alphabk(),
                static_cast<uint32_t>(key_lvl0[i])));
        }
    }

    template <class RandomEngine>
    static cloud_key make(RandomEngine &rand, const secret_key<P> &key)
    {
        return cloud_key{rand, key.key_lvl0(), key.key_lvl1()};
    }

    const key_switching_key<P> &ksk() const
    {
        return *ksk_;
    }
    const std::vector<trgswfft_lvl1<P>> &bkfftlvl01() const
    {
        return bkfftlvl01_;
    }

    tlwe_lvl0<P> nand(const tlwe_lvl0<P> &lhs, const tlwe_lvl0<P> &rhs) const;
};

template <class P>
class tlwe_lvl1 {
public:
    using container = detail::nd_array<uint32_t, P::N() + 1>;

private:
    container data_;

public:
    explicit tlwe_lvl1(container data) : data_(std::move(data))
    {
    }

    tlwe_lvl0<P> identity_key_switch(const key_switching_key<P> &ksk) const;
};

template <class P>
class trgswfft_lvl1 {
public:
    using container = detail::nd_array<double, 2 * P::l(), 2, P::N()>;

private:
    container data_;

public:
    trgswfft_lvl1(container data) : data_(std::move(data))
    {
    }

    template <class RandomEngine>
    static trgswfft_lvl1 sym_encrypt(
        RandomEngine &rand,
        const typename secret_key<P>::container_key_lvl1 &key_lvl1,
        double alpha, uint32_t plain);

    detail::const_array_slice<double, P::N()> slice(size_t i, size_t j) const
    {
        return data_.slice(i, j);
    }
};

template <class P>
class trlwe_lvl1 {
    template <class RandomEngine>
    friend trgswfft_lvl1<P> trgswfft_lvl1<P>::sym_encrypt(
        RandomEngine &, const typename secret_key<P>::container_key_lvl1 &,
        double, uint32_t);
    friend tlwe_lvl1<P> tlwe_lvl0<P>::gate_bootstrapping_to_lvl1(
        const std::vector<trgswfft_lvl1<P>> &) const;

public:
    using container = detail::nd_array<uint32_t, P::N()>;

private:
    container poly0_, poly1_;

public:
    trlwe_lvl1()
    {
    }

    explicit trlwe_lvl1(container poly0, container poly1)
        : poly0_(std::move(poly0)), poly1_(std::move(poly1))
    {
    }

    template <class RandomEngine>
    static trlwe_lvl1 sym_encrypt_zero(
        RandomEngine &rand,
        const typename secret_key<P>::container_key_lvl1 &key, double alpha)
    {
        std::uniform_int_distribution<uint32_t> torus(
            0, std::numeric_limits<uint32_t>::max());
        std::normal_distribution<double> gaussian(0.0, alpha);

        container poly0;
        for (uint32_t &v : poly0)
            v = torus(rand);

        container poly1 = poly_mul_lvl1(poly0, key);
        for (uint32_t &v : poly1)
            v += detail::double2torus32(gaussian(rand));

        return trlwe_lvl1{poly0, poly1};
    }

    template <class RandomEngine>
    static trlwe_lvl1 sym_encrypt(
        RandomEngine &rand,
        const typename secret_key<P>::container_key_lvl1 &key, double alpha,
        detail::nd_array<bool, P::N()> &plain)
    {
        trlwe_lvl1 ret = sym_encrypt_zero(rand, key, alpha);
        for (size_t i = 0; i < P::N(); i++)
            ret.poly1_[i] += plain[i] ? P::mu() : -P::mu();
        return ret;
    }

    detail::nd_array<bool, P::N()> sym_decrypt(
        const typename secret_key<P>::container_key_lvl1 &key) const
    {
        container mul = poly_mul_lvl1(poly0_, key);

        container phase = poly1_;
        for (size_t i = 0; i < P::N(); i++)
            phase[i] -= mul[i];

        detail::nd_array<bool, P::N()> ret;
        for (size_t i = 0; i < P::N(); i++)
            ret[i] = static_cast<int32_t>(phase[i]) > 0;

        return ret;
    }

    void polynomial_mul_by_xai_minus_one(trlwe_lvl1 &out, uint32_t a) const
    {
        polynomial_mul_by_xai_minus_one_impl(out.poly0_, a, poly0_);
        polynomial_mul_by_xai_minus_one_impl(out.poly1_, a, poly1_);
    }

    tlwe_lvl1<P> sample_extract_index(uint32_t index) const
    {
        typename tlwe_lvl1<P>::container ret;
        for (size_t i = 0; i <= index; i++)
            ret[i] = poly0_[index - i];
        for (size_t i = index + 1; i < P::N(); i++)
            ret[i] = -poly0_[P::N() + index - i];
        ret[P::N()] = poly1_[index];

        return tlwe_lvl1<P>{std::move(ret)};
    }

    decomposed_trlwe_in_fd_lvl1<P> decomposition_and_ifft(
        detail::nd_array<double, 2 * P::l(), P::N()> &decvecfft) const;
    void trgswfft_external_product(trlwe_lvl1<P> &out,
                                   const trgswfft_lvl1<P> &trgswfft) const;

private:
    static container poly_mul_lvl1(const container &a, const container &b)
    {
        thread_local detail::fft_processor<P::N()> fftproc;
        const detail::nd_array<double, P::N()> ffta =
                                                   fftproc.twist_ifft_lvl1(a),
                                               fftb =
                                                   fftproc.twist_ifft_lvl1(b);
        return fftproc.twist_fft_lvl1(
            detail::mul_in_fd(ffta.slice(), fftb.slice()));
    }

    static void polynomial_mul_by_xai_minus_one_impl(container &out, uint32_t a,
                                                     const container &poly)
    {
        if (a == 0) {
            out = poly;
            return;
        }

        constexpr uint32_t N = P::N();
        if (a < N) {
            for (size_t i = 0; i < a; i++)
                out[i] = -poly[i - a + N] - poly[i];
            for (size_t i = a; i < N; i++)
                out[i] = poly[i - a] - poly[i];
        }
        else {
            const uint32_t aa = a - N;
            for (size_t i = 0; i < aa; i++)
                out[i] = poly[i - aa + N] - poly[i];
            for (size_t i = aa; i < N; i++)
                out[i] = -poly[i - aa] - poly[i];
        }
    }
};

template <class P>
class tlwe_lvl0 {
public:
    using container = detail::nd_array<uint32_t, P::n() + 1>;

private:
    container data_;

public:
    explicit tlwe_lvl0(container data) : data_(std::move(data))
    {
    }

    template <class RandomEngine>
    static tlwe_lvl0 boots_sym_encrypt(
        RandomEngine &rand,
        const typename secret_key<P>::container_key_lvl0 &key, bool plain)
    {
        return sym_encrypt(rand, key, P::alpha(), plain ? P::mu() : -P::mu());
    }

    template <class RandomEngine>
    static tlwe_lvl0 sym_encrypt(
        RandomEngine &rand,
        const typename secret_key<P>::container_key_lvl0 &key, double alpha,
        uint32_t plain)
    {
        container src;
        sym_encrypt(src.begin(), rand, key, alpha, plain);
        return tlwe_lvl0{std::move(src)};
    }

    template <class RandomEngine>
    static void sym_encrypt(
        detail::array_slice<uint32_t, P::n() + 1> out, RandomEngine &rand,
        const typename secret_key<P>::container_key_lvl0 &key, double alpha,
        uint32_t plain)
    {
        std::uniform_int_distribution<uint32_t> torus(
            0, std::numeric_limits<uint32_t>::max());
        std::normal_distribution<double> gaussian(0.0, alpha);

        out[P::n()] = plain + detail::double2torus32(gaussian(rand));
        for (size_t i = 0; i < P::n(); i++) {
            out[i] = torus(rand);
            out[P::n()] += out[i] * key[i];
        }
    }

    bool boots_sym_decrypt(
        const typename secret_key<P>::container_key_lvl0 &key) const
    {
        uint32_t phase = data_[P::n()];
        for (size_t i = 0; i < P::n(); i++)
            phase -= data_[i] * key[i];
        return static_cast<int32_t>(phase) > 0;
    }

    tlwe_lvl0 gate_bootstrapping(
        const std::vector<trgswfft_lvl1<P>> &bkfftlvl01,
        const key_switching_key<P> &ksk) const
    {
        return gate_bootstrapping_to_lvl1(bkfftlvl01).identity_key_switch(ksk);
    }

    tlwe_lvl1<P> gate_bootstrapping_to_lvl1(
        const std::vector<trgswfft_lvl1<P>> &bkfftlvl01) const
    {
        constexpr uint32_t n = P::n(), N = P::N();

        uint32_t bara =
            2 * N - detail::mod_switch_from_torus32(2 * N, data_[n]);
        trlwe_lvl1<P> acc = rotated_test_vector(bara), tmp0, tmp1;
        for (size_t i = 0; i < n; i++) {
            bara = detail::mod_switch_from_torus32(2 * N, data_[i]);
            if (bara == 0)
                continue;
            acc.polynomial_mul_by_xai_minus_one(/* out */ tmp0, bara);
            tmp0.trgswfft_external_product(/* out */ tmp1, bkfftlvl01[i]);
            for (size_t i = 0; i < N; i++) {
                acc.poly0_[i] += tmp1.poly0_[i];
                acc.poly1_[i] += tmp1.poly1_[i];
            }
        }

        return acc.sample_extract_index(0);
    }

    tlwe_lvl0 nand(const tlwe_lvl0 &rhs,
                   const std::vector<trgswfft_lvl1<P>> &bkfftlvl01,
                   const key_switching_key<P> &ksk) const
    {
        container ret;
        for (size_t i = 0; i <= P::n(); i++)
            ret[i] = -data_[i] - rhs.data_[i];
        ret[P::n()] += 1U << 29;

        return tlwe_lvl0{std::move(ret)}.gate_bootstrapping(bkfftlvl01, ksk);
    }

private:
    trlwe_lvl1<P> rotated_test_vector(uint32_t bara) const
    {
        constexpr uint32_t N = P::N(), mu = P::mu();

        typename trlwe_lvl1<P>::container poly1;

        if (bara < N) {
            for (uint32_t i = 0; i < bara; i++)
                poly1[i] = -mu;
            for (uint32_t i = bara; i < N; i++)
                poly1[i] = mu;
        }
        else {
            const uint32_t baraa = bara - N;
            for (uint32_t i = 0; i < baraa; i++)
                poly1[i] = mu;
            for (uint32_t i = baraa; i < N; i++)
                poly1[i] = -mu;
        }

        return trlwe_lvl1<P>{typename trlwe_lvl1<P>::container{{}},
                             std::move(poly1)};
    }
};

template <class P>
tlwe_lvl0<P> cloud_key<P>::nand(const tlwe_lvl0<P> &lhs,
                                const tlwe_lvl0<P> &rhs) const
{
    return lhs.nand(rhs, bkfftlvl01_, *ksk_);
}

template <class P>
template <class RandomEngine>
key_switching_key<P>::key_switching_key(
    RandomEngine &rand,
    const typename secret_key<P>::container_key_lvl0 &key_lvl0,
    const typename secret_key<P>::container_key_lvl1 &key_lvl1)
{
    for (size_t i = 0; i < P::N(); i++) {
        for (size_t j = 0; j < P::t(); j++) {
            for (size_t k = 0; k < (1u << P::basebit()) - 1; k++) {
                uint32_t val = key_lvl1[i] * (k + 1) *
                               (1U << (32 - (j + 1) * P::basebit()));
                tlwe_lvl0<P>::sym_encrypt(data_.slice(i, j, k), rand, key_lvl0,
                                          P::alphaks(), val);
            }
        }
    }
}

template <class P>
tlwe_lvl0<P> tlwe_lvl1<P>::identity_key_switch(
    const key_switching_key<P> &ksk) const
{
    const uint32_t prec_offset = 1U << (32 - (1 + P::basebit() * P::t()));
    const uint32_t mask = (1U << P::basebit()) - 1;

    typename tlwe_lvl0<P>::container ret = {};
    ret[P::n()] = data_[P::N()];
    for (size_t i = 0; i < P::N(); i++) {
        uint32_t aibar = data_[i] + prec_offset;
        for (size_t j = 0; j < P::t(); j++) {
            uint32_t aij = (aibar >> (32 - (j + 1) * P::basebit())) & mask;
            if (aij == 0)
                continue;
            for (size_t k = 0; k <= P::n(); k++)
                ret[k] -= ksk(i, j, aij - 1, k);
        }
    }

    return tlwe_lvl0<P>{std::move(ret)};
}

template <class P>
template <class RandomEngine>
trgswfft_lvl1<P> trgswfft_lvl1<P>::sym_encrypt(
    RandomEngine &rand,
    const typename secret_key<P>::container_key_lvl1 &key_lvl1, double alpha,
    uint32_t plain)
{
    // trgsw sym encrypt
    // FIXME: more efficient implementation?
    std::vector<trlwe_lvl1<P>> trgsw;
    trgsw.reserve(2 * P::l());
    for (size_t i = 0; i < 2 * P::l(); i++)
        trgsw.push_back(trlwe_lvl1<P>::sym_encrypt_zero(rand, key_lvl1, alpha));
    for (uint32_t i = 0; i < P::l(); i++) {
        uint32_t h = 1U << (32 - (i + 1) * P::Bgbit());
        trgsw[i].poly0_[0] += plain * h;
        trgsw[i + P::l()].poly1_[0] += plain * h;
    }

    // trgswfft sym encrypt
    thread_local detail::fft_processor<P::N()> fftproc;
    container trgswfft;
    for (uint32_t i = 0; i < 2 * P::l(); i++) {
        fftproc.twist_ifft_lvl1(trgswfft.slice(i, 0), trgsw[i].poly0_.slice());
        fftproc.twist_ifft_lvl1(trgswfft.slice(i, 1), trgsw[i].poly1_.slice());
    }

    return trgswfft_lvl1{std::move(trgswfft)};
}

template <class P>
decomposed_trlwe_in_fd_lvl1<P> trlwe_lvl1<P>::decomposition_and_ifft(
    detail::nd_array<double, 2 * P::l(), P::N()> &decvecfft) const
{
    constexpr uint32_t Bgbit = P::Bgbit(), N = P::N(), l = P::l(), Bg = P::Bg();

    // offsetgen
    uint32_t offset = 0;
    for (size_t i = 1; i <= l; i++)
        offset += Bg / 2 * (1U << (32 - i * Bgbit));

    // decomposition
    constexpr uint32_t mask = static_cast<uint32_t>((1UL << Bgbit) - 1);
    detail::nd_array<uint32_t, 2 * l, N> decvec;
    for (size_t i = 0; i < N; i++) {
        decvec(0, i) = poly0_[i] + offset;
        decvec(l, i) = poly1_[i] + offset;
    }

    constexpr uint32_t halfBg = (1UL << (Bgbit - 1));
    for (int i = l - 1; i >= 0; i--) {
        for (size_t j = 0; j < N; j++) {
            decvec(i, j) =
                ((decvec(0, j) >> (32 - (i + 1) * Bgbit)) & mask) - halfBg;
            decvec(i + l, j) =
                ((decvec(l, j) >> (32 - (i + 1) * Bgbit)) & mask) - halfBg;
        }
    }

    // twist ifft
    thread_local detail::fft_processor<N> fftproc;
    for (size_t i = 0; i < 2 * l; i++)
        fftproc.twist_ifft_lvl1(decvecfft.slice(i), decvec.slice(i));

    return decomposed_trlwe_in_fd_lvl1<P>{decvecfft};
}

template <class P>
void trlwe_lvl1<P>::trgswfft_external_product(
    trlwe_lvl1<P> &out, const trgswfft_lvl1<P> &trgswfft) const
{
    constexpr uint32_t Bgbit = P::Bgbit(), N = P::N(), l = P::l(), Bg = P::Bg();
    thread_local detail::fft_processor<N> fftproc;

    ///// offset gen/////
    uint32_t offset = 0;
    for (size_t i = 1; i <= l; i++)
        offset += Bg / 2 * (1U << (32 - i * Bgbit));

    ///// decomposition /////
    constexpr uint32_t mask = static_cast<uint32_t>((1UL << Bgbit) - 1);
    constexpr uint32_t halfBg = (1UL << (Bgbit - 1));
    detail::nd_array<uint32_t, 2 * l, N> decvec;
    for (size_t i = 0; i < N; i++) {
        decvec(0, i) = poly0_[i] + offset;
        decvec(l, i) = poly1_[i] + offset;
    }
    for (int i = l - 1; i >= 0; i--) {
        for (size_t j = 0; j < N; j++) {
            decvec(i, j) =
                ((decvec(0, j) >> (32 - (i + 1) * Bgbit)) & mask) - halfBg;
            decvec(i + l, j) =
                ((decvec(l, j) >> (32 - (i + 1) * Bgbit)) & mask) - halfBg;
        }
    }

    ///// ifft /////
    detail::nd_array<double, 2 * P::l(), P::N()> decvecfft;
    for (size_t i = 0; i < 2 * l; i++)
        fftproc.twist_ifft_lvl1(decvecfft.slice(i), decvec.slice(i));

    ///// mul and fma in fd /////
    detail::nd_array<double, P::N()> poly0 =
        detail::mul_in_fd<P::N()>(decvecfft.slice(0), trgswfft.slice(0, 0));
    detail::nd_array<double, P::N()> poly1 =
        detail::mul_in_fd<P::N()>(decvecfft.slice(0), trgswfft.slice(0, 1));
    for (size_t i = 1; i < 2 * P::l(); i++) {
        detail::fma_in_fd(poly0.slice(), decvecfft.cslice(i),
                          trgswfft.slice(i, 0));
        detail::fma_in_fd(poly1.slice(), decvecfft.cslice(i),
                          trgswfft.slice(i, 1));
    }

    ///// twsit fft /////
    fftproc.twist_fft_lvl1(/* out */ out.poly0_.slice(), poly0.slice());
    fftproc.twist_fft_lvl1(/* out */ out.poly1_.slice(), poly1.slice());
}

}  // namespace aqtfhe2

#endif
