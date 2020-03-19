#include "aqtfhe2.hpp"
//
#include "aqtfhe2.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

//#include <gperftools/profiler.h>

namespace {

template <size_t N, class RandomEngine>
aqtfhe2::detail::nd_array<bool, N> random_bool_array(RandomEngine& rand)
{
    aqtfhe2::detail::nd_array<bool, N> ret;
    std::binomial_distribution<int> binary;
    for (bool& v : ret)
        v = binary(rand);
    return ret;
}

#ifndef NDEBUG
void test_nd_array()
{
    using namespace aqtfhe2;
    using namespace aqtfhe2::detail;

    {
        nd_array<int, 6, 5, 4> ary = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        };
        for (size_t i = 0; i < 6; i++)
            for (size_t j = 0; j < 5; j++)
                for (size_t k = 0; k < 4; k++)
                    if (i == 4 && j == 1 && k == 1)
                        assert(ary(i, j, k) == 1);
                    else
                        assert(ary(i, j, k) == 0);
    }
    {
        constexpr int hoge = nd_array<int, 1, 1, 1>{1}(0, 0, 0);
        assert(hoge == 1);
    }
}

void test_tlwe_lvl0()
{
    using secret_key = aqtfhe2::secret_key<aqtfhe2::params::CGGI16>;
    using tlwe_lvl0 = aqtfhe2::tlwe_lvl0<aqtfhe2::params::CGGI16>;

    {
        std::mt19937 rand;
        auto sk = secret_key::make(rand);

        for (int i = 0; i < 10; i++) {
            bool plain = random_bool_array<1>(rand)[0];
            auto e = tlwe_lvl0::boots_sym_encrypt(rand, sk.key_lvl0(), plain);
            assert(plain == e.boots_sym_decrypt(sk.key_lvl0()));
        }
    }
}

void test_trlwe_lvl1()
{
    using P = aqtfhe2::params::CGGI16;
    using secret_key = aqtfhe2::secret_key<P>;
    using trlwe_lvl1 = aqtfhe2::trlwe_lvl1<P>;

    {
        std::mt19937 rand;

        auto sk = secret_key::make(rand);

        for (int i = 0; i < 10; i++) {
            auto plain = random_bool_array<P::N()>(rand);
            auto trlwe = trlwe_lvl1::sym_encrypt(rand, sk.key_lvl1(), plain);
            assert(plain == trlwe.sym_decrypt(sk.key_lvl1()));
        }
    }
}

void test_trgswfft_external_product()
{
    using P = aqtfhe2::params::CGGI16;
    using secret_key = aqtfhe2::secret_key<P>;
    using trgswfft_lvl1 = aqtfhe2::trgswfft_lvl1<P>;
    using trlwe_lvl1 = aqtfhe2::trlwe_lvl1<P>;

    {
        std::mt19937 rand;
        auto sk = secret_key::make(rand);

        for (int i = 0; i < 10; i++) {
            auto plain = random_bool_array<P::N()>(rand);
            auto trlwe = trlwe_lvl1::sym_encrypt(rand, sk.key_lvl1(), plain);
            auto trgswfft = trgswfft_lvl1::sym_encrypt(rand, sk.key_lvl1(), 1);
            trlwe_lvl1 res;
            trlwe.trgswfft_external_product(/* out */ res, trgswfft);
            assert(plain == res.sym_decrypt(sk.key_lvl1()));
        }
    }
}

void test_bootstrapping()
{
    using P = aqtfhe2::params::CGGI16;
    using secret_key = aqtfhe2::secret_key<P>;
    using cloud_key = aqtfhe2::cloud_key<P>;
    using tlwe_lvl0 = aqtfhe2::tlwe_lvl0<P>;

    {
        std::mt19937 rand;
        auto sk = secret_key::make(rand);
        auto ck = cloud_key::make(rand, sk);

        for (int i = 0; i < 10; i++) {
            auto plain = random_bool_array<1>(rand)[0];
            auto tlwe =
                tlwe_lvl0::boots_sym_encrypt(rand, sk.key_lvl0(), plain);
            auto booted_tlwe =
                tlwe.gate_bootstrapping(ck.bkfftlvl01(), ck.ksk());
            assert(plain == booted_tlwe.boots_sym_decrypt(sk.key_lvl0()));
        }
    }
}

void test_nand()
{
    using P = aqtfhe2::params::CGGI16;
    using secret_key = aqtfhe2::secret_key<P>;
    using cloud_key = aqtfhe2::cloud_key<P>;

    {
        std::mt19937 rand;
        auto sk = secret_key::make(rand);
        auto ck = cloud_key::make(rand, sk);

        for (int i = 0; i < 10; i++) {
            auto plain = random_bool_array<2>(rand);
            auto enc0 = sk.encrypt(rand, plain[0]),
                 enc1 = sk.encrypt(rand, plain[1]);
            assert(!(plain[0] & plain[1]) == sk.decrypt(ck.nand(enc0, enc1)));
        }
    }
}
#endif

}  // namespace

// For simplicity
using P = aqtfhe2::params::CGGI16;
using secret_key = aqtfhe2::secret_key<P>;
using cloud_key = aqtfhe2::cloud_key<P>;
using encrypted_bit = aqtfhe2::tlwe_lvl0<P>;
template <class T, size_t... Shape>
using nd_array = aqtfhe2::detail::nd_array<T, Shape...>;

int main()
{
#ifndef NDEBUG
    test_nd_array();
    test_tlwe_lvl0();
    test_trlwe_lvl1();
    test_trgswfft_external_product();
    test_bootstrapping();
    test_nand();
#else
    std::random_device rand;
    auto sk = secret_key::make(rand);
    auto ck = cloud_key::make(rand, sk);

    const size_t TEST_SIZE = 1000;

    // Prepare plain data
    nd_array<bool, TEST_SIZE> p = random_bool_array<TEST_SIZE>(rand),
                              q = random_bool_array<TEST_SIZE>(rand), r;
    for (size_t i = 0; i < TEST_SIZE; i++)
        r[i] = !(p[i] & q[i]);

    // Encrypt the data
    std::vector<encrypted_bit> p_enc, q_enc, r_enc;
    for (bool v : p)
        p_enc.push_back(sk.encrypt(rand, v));
    for (bool v : q)
        q_enc.push_back(sk.encrypt(rand, v));
    r_enc.reserve(TEST_SIZE);

    // Calc NAND
    auto begin = std::chrono::high_resolution_clock::now();
    // ProfilerStart("sample.prof");
    for (size_t i = 0; i < TEST_SIZE; i++)
        r_enc.push_back(ck.nand(p_enc[i], q_enc[i]));
    // ProfilerStop();
    auto end = std::chrono::high_resolution_clock::now();

    // Print elapsed time.
    auto usec =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cerr << usec.count() / TEST_SIZE << "us / gate" << std::endl;

    // Check the results
    for (size_t i = 0; i < TEST_SIZE; i++)
        assert(r[i] == sk.decrypt(r_enc[i]));
#endif

    return 0;
}
