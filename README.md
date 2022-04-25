# simd_granodi.h

### Easy 128-bit AArch64 NEON / x64 SSE2 SIMD in a single header file, for both plain C99, and C++ with operator overloading

Currently has full test coverage, but not used in production yet. Every intrinsic/function additionally has a "generic" non-vector implementation so your code can compile on platforms without SSE/NEON (or can be forced with `SSE_GRANODI_FORCE_GENERIC`).

The C++ classes are written in terms of the C implementation.

`simd_granodi.h` is the only file you need.
`simd_granodi_sse2_on_neon.h` contains some macros used by the author of this project, but not needed in new projects.
`test_simd_granodi.c` contains the test code.

## Example code (C++)

```cpp
#include <iostream>
#include "simd_granodi.h"

using namespace simd_granodi;

template <typename T>
T square_vec(const T& x) { return x * x; }

// Very approximate log2()
template <typename VecType>
VecType log2_p3(const VecType& x) {
    // VecType::from() static method aids templated code
    VecType exponent = VecType::from(x.exponent_s32()),
        mantissa = x.mantissa();

    // Uses FMA if available on hardware
    // 2nd argument is implicitly constructed as a vector
    mantissa = mantissa.mul_add(1.6404256133344508e-1, -1.0988652862227437)
        .mul_add(mantissa, 3.1482979293341158)
        .mul_add(mantissa, -2.2134752044448169);

    // This would also work:
    // mantissa = (((1.6404256133344508e-1*mantissa - 1.0988652862227437)
    //     *mantissa + 3.1482979293341158)*mantissa - 2.2134752044448169);

    // Branchless conditional using implicitly constructed Compare type
    // Note: std::log2() returns NaN for x < 0, and -inf for x == 0
    return (x > 0.0).choose(exponent + mantissa, VecType::minus_infinity());
}

int main() {
    float square_f = square_vec(6.0f);

    // Vec_f64x1 is a "shim" for double, needed as this function uses
    // class methods
    auto log2_d = log2_p3(Vec_f64x1{10.0});

    // For 4 floats (packed single) or 2 doubles (packed double)
    auto square_4f = log2_p3(Vec_ps{4, 3, 9, 2});
    auto log2_2d = square_vec(Vec_pd{7, 3});

    std::cout << "6 squared is " << square_f << std::endl <<
    "approx log2(10) is " << log2_d.data() << std::endl <<

    "approx log2() of {4, 3, 9, 2} is {" << square_4f.f3() << ", " <<
    square_4f.f2() << ", " << square_4f.f1() << ", " <<
    square_4f.f0() << "}\n" <<

    "{7, 3} squared is {" << log2_2d.d1() << ", " << log2_2d.d0() << "}\n";
}
```
