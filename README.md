# aqTFHE2

Yet another implementation of TFHE in C++17.

About 7.7 ms/gate on Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz.

## Simple Usage

Write code like this:

```cpp
#include "aqtfhe2.hpp"

int main()
{
    using P = aqtfhe2::params::CGGI16;
    std::random_device rand;

    auto sk = aqtfhe2::secret_key<P>::make(rand);
    auto ck = aqtfhe2::cloud_key<P>::make(rand, sk);

    auto e0 = sk.encrypt(rand, false), e1 = sk.encrypt(rand, true);

    auto eres = ck.nand(e0, e1);

    assert(sk.decrypt(eres) == true);
}
```

And compile it:

```
$ clang++ -std=c++17 -march=native -O3 hoge.cpp -I spqlios/ -L spqlios/build/ -lspqlios
$ ./a.out
```

## Performance

About 7.7 ms/gate for old parameter [CGGI16], and 12.6 ms/gate for new parameter [CGGI19].
Both are measured on Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz.

You can verify it by using `main.cpp`:

```
$ make     # Compile main.cpp
$ ./main   # Run
Old parameter [CGGI16]:	7650us / gate
New parameter [CGGI19]:	12570us / gate
```

See [TFHE's web site](https://tfhe.github.io/tfhe/security_and_params.html)
for the details about old and new parameters.

## Caveat

Some functions in aqTFHE2 need a random number generator as argument.
**Use `std::random_device` there** if you don't care about it,
because it is supposed to be the only cryptographically secure
pseudo-random number generator (CSPRNG) in C++ standard library
(See [here](https://timsong-cpp.github.io/cppwp/n4659/rand) for the details).

## Licenses

This project is licensed under Apache License Version 2.0.
See the file LICENSE.

However the directory `spqlios/` is not my work but [TFHE](https://tfhe.github.io/tfhe/)'s one.
See the file `spqlios/LICENSE`.

## References

- [TFHE](https://tfhe.github.io/tfhe/)
- [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp)
    - aqTFHE2 is strongly inspired by TFHEpp.
- [aqTFHE](https://github.com/ushitora-anqou/aqTFHE)
