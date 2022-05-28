# simd_granodi.h

### Easy C/C++ SIMD in a single header file, supporting AArch64 NEON and x86_64 SSE2

## Features
- Supports C99, and C++11 with extensive operator overloading, type traits, templated conversions etc.
- Full test coverage.
- Type-safe C++ comparison and bit-masking.
- The C++ classes are written in terms of the C implementation.
- The C++ classes are (mostly) immutable, apart from the `+=`, `-=`, `*=`, `/=`, `&=`, `|=`, `^=` operators. Most non-static methods are marked `const`.
- The C implementation mostly uses macros (where safe), for faster debug builds. In addition, C code should take almost the same amount of time to compile as if you had used the intrinsics directly.
- The included generic, non-vector implementation allows for compilation on any target. (This implementation can be forced by defining `SIMD_GRANODI_FORCE_GENERIC`). It also serves as documentation for those not familiar with SIMD intrinsics.
- Avoids undefined behaviour.
- Tested on GCC, Clang, and MSVC++.
- `simd_granodi.h` is the only file you need.
- Designed to be easy enough to use for someone who has never done any SIMD programming before.

### Why 64-bit only? Why only SSE2?

This library was written so that the author could easily write cross-platform Audio DSP code that runs on all x86_64 machines, and newer AArch64 machines (ie the latest hardware from Apple). It targets 128-bit SIMD types (32x4 or 64x2).

The x86\_64 implementation limits itself to SSE2, as many otherwise-capable modern low-end CPUs do not support AVX, SSE2 is guaranteed to be implemented on all x86_64 machines, and later SSE instructions (eg SSE 4.2) only provide a marginal speed improvement to the functionality of this library. Also, NEON only supports 128-bit vectors. However, it would be very easy to add AVX support in future, and simply wrap 2 or 4 128-bit registers on NEON.

### Slow / non-vector functions

Some platforms do not have intrinsic functions for some SIMD operations, and so they are implemented generically and may be slower. A list of these functions/macros, per-platform, is contained in a comment at the start of the `simd_granodi.h` file. If you are using the C++ classes, you may wish to search for those names in the file to see which methods they correspond to.

## C++ documentation
### Namespaces

All of the C++ code is inside the namespace `simd_granodi`. All of the code examples below assume you are `using namespace simd_granodi;`, but you may choose to do something like `namespace sg = simd_granodi;`.

The C functions / macros are **not** inside a namespace (because of the use of macros), but typically have the prefix `sg_`.

### Fundamental types
All of the following vector types are 128-bit in size:

- `Vec_pi32` - Vector of 4 **32**-bit signed **p**acked **i**ntegers. AKA `Vec_s32x4`
- `Vec_pi64` - Vector of 2 **64**-bit signed **p**acked **i**ntegers. AKA `Vec_s64x2`
- `Vec_ps` - Vector of 4 **p**acked **s**ingle-precision floats. AKA `Vec_f32x4`
- `Vec_pd` - Vector of 2 **p**acked **d**ouble-precision floats. AKA `Vec_f64x2`

The following are "scalar wrapper" types, which allow you to write templated code that operates either on vectors or C++ built-in types:

- `Vec_s32x1` - Wrapper for `int32_t`
- `Vec_s64x1` - Wrapper for `int64_t`
- `Vec_f32x1` - Wrapper for `float`. AKA `Vec_ss`
- `Vec_f64x1` - Wrapper for `double`. AKA `Vec_sd`

### Comparison types

The following are type-safe comparison-types (implemented as bit-masks) that arise as the result of comparing two vectors:

- `Compare_pi32` - Result of comparing two `Vec_pi32`. AKA `Compare_s32x4`
- `Compare_pi64` - Result of comparing two `Vec_pi64`. AKA `Compare_s64x2`
- `Compare_ps` - Result of comparing two `Vec_ps`. AKA `Compare_f32x4` 
- `Compare_pd` - Result of comparing two `Vec_pd`. AKA `Compare_f64x2`

The following are type-safe comparison types for the equivalent "scalar wrapper" types, allowing you to write templated code that operates either on vectors or C++ built-in types. They are a simple wrapper for `bool`, and should get optimized out of existence when used:

- `Compare_s32x1` - Result of comparing two `Vec_s32x1`
- `Compare_s64x1` - Result of comparing two `Vec_s64x1`
- `Compare_f32x1` - Result of comparing two `Vec_f32x1`. AKA `Compare_ss`
- `Compare_s64x1` - Result of comparing two `Vec_f64x1`. AKA `Compare_sd`

### Constructors

#### Default constructor

- All vector types are default-constructed to hold a value of zero. This is for safety and convenience, and this assignment typically gets optimized out.
- All comparison types are default-constructed to hold a value of `false`. With vector comparisons, this is a bit-mask comprised of all zeros. With "scalar wrapper" comparisons, this is a `bool` with value `false`.

#### "Broadcast" constructor

- All vector types have a "broadcast" constructor that takes a single value and "broadcasts" it to all elements of the vector. For example, `Vec_ps{3.0f}` will result in the vector `Vec_ps{3.0f, 3.0f, 3.0f, 3.0f}`. This also allows for convenient arithmetic with constants, for example `Vec_ps{3.0f} + 1.0f` will give the vector `{4.0f, 4.0f, 4.0f, 4.0f}` as the `1.0f` is implicitly constructed into a `Vec_ps`.
- Note that vector types do **not** have a broadcast constructor that takes an equivalent "scalar wrapper" type as an argument, due to constructor overload ambiguity when using literals/constants in code. Ie you can **not** do `Vec_ps{Vec_f32x1{1.0f}}`. However there are easy workarounds for this.
- All comparison types also have a "broadcast" constructor which accepts a `bool`. For vector comparisons, **all** of the bits of the bit-mask are set to `0` if this is `false`, or `1` if this is true. For the "scalar wrapper" types, this sets the value of the `bool` member.

#### "Vector" constructor

- All Vector types can be constructed by specifying the value of each element. Following the SSE2 convention, the elements are specified in reverse order. For example, `Vec_ps{3.0f, 2.0f, 1.0f, 0.0f}` creates a vector with the value 3 at index 3, value 2 at index 2, value 1 at index 1, and value 0 at index 0. When these values are compile-time constants, this constructor is typically optimized into a single instruction, otherwise it may take several instructions.
- All comparison types also have a "vector" constructor that takes a `bool` to specify a bit-mask at each element. `true` will be interpreted as **all** bits set to 1, and `false` will be interpreted as all bits set to 0. The reverse ordering convention is the same as for vector types.

### Element access

The elements of a vector can be accessed with the templated `.get<int32_t>()` method. The index is passed as the template argument. An out of range index will cause a compile time error. Example: `Vec_pd{4.0, 3.0}.get<1>()` will return a `double` with value `4.0`.

Every vector and comparison type also has a `.data()` method that allows access to the underlying representation of that type.

### Arithmetic operators

All `Vec_` types implement the following standard arithmetic operators: `+=`, `+`, `-=`, `-`, `*=`, `*`, `/=`, `/`. Also, integer types support both the pre- and postfix `++` and `--` operators.

### Bitwise operators

All vector types, including floating-point types, implement the following bitwise operators: `&=`, `&`, `|=`, `|`, `^=`, `^`, `~`.

**Note:** You might assume that, since `Compare_` types use bitwise operations to mask or "select" a result, that they also implement these bitwise operators. However, for reasons of type safety, they do not implement these operators at all. Instead, they only implement logical operators.

### Comparison operators for `Vec_` types

When a comparison operator is used with two `Vec_` types, it returns a result of the corresponding `Compare_` type. All vector types implement the following comparison operators: `<`, `<=`, `==`, `!=`, `>=`, `>`.

### Logical operators for `Compare_` types

All `Compare_` types support the following logical operators: `&&`, `||`, `!`.

### Comparison operators for `Compare_` types

All `Compare_` types support the following comparison operators: `==`, `!=`.

### `Compare_` choose / selection methods

All `Compare_` types have two important methods: `.choose(v_true, v_false)` and `.choose_else_zero(v_true)`. `v_true` and `v_false` must be of the `Vec_` type that corresponds to the `Compare_` type. These methods return a `Vec_` by selection.

This is best explained by example: `(Vec_ps{3.0f} < 2.0f).choose(7.0f, 8.0f)` will return `Vec_ps{8.0f}`, because 3 is not smaller than 2.

The `.choose_else_zero()` methods are a common optimization of `.choose()`. Whereas `.choose()` typically takes four CPU instructions, `.choose_else_zero()` only takes one. Using the example above, `(Vec_ps{3.0f} < 2.0f).choose_else_zero(7.0f)` would return `Vec_ps{0.0f}`.

#### An important note on `.choose()` method implementation

For the 128-bit vector types, the `.choose()` and `.choose_else_zero()` methods simply use bit-masking. The advantage is that this is completely branch-less, but the disadvantage is that both "branches" or possibilities are calculated: the unneeded result is then discarded.

For the "scalar wrapper" types, the `.choose()` and `.choose_else-zero()` methods may also appear to calculate both "branches", and in fact they will do so in unoptimized builds. But, with any modern optimizing compiler, these methods will get inlined and the compiler will generate a conditional jump, so usually only one "branch" is calculated.

### Bit-shifting

All signed integer types implement bit-shifting methods. These are not implemented for floating point types. If you wish to shift by an immediate value (compile-time constant), you can use one of the following methods where `amount` must be a compile-time constant of type `int32_t`:

- `.shift_l_imm<amount>()` - Return a new vector with each element shifted left by `amount`.
- `.shift_rl_imm<amount>()` - Return a new vector with each element shifted right logically by `amount`.
- `.shift_ra_imm<amount>()` - Return a new vector with each element shifted right arithmetically by `amount`.

For shifting by an amount determined at run-time, by another `Vec_` of the same type, these are:

- `.shift_l(const Vec_& amount)` - Return a new vector with each element shifted left by the corresponding element in `amount`.
- `.shift_rl(const Vec_& amount)` - Return a new vector with each element shifted right logically by the corresponding element in `amount`.
- `.shift_ra(const Vec_& amount)` - Return a new vector with each element shifted right arithmetically by the corresponding element in `amount`.

### Shuffling (rearranging vectors internally)

All `Vec_` types with more than one element have a templated `.shuffle<>()` method that takes either 2 or 4 template parameters of type `int32_t` and returns a new vector with its internal elements rearranged. The template parameters represent the source indexes for the new vector, and will fail to compile if they are out of range.

This is best explained via example:

- `Vec_ps{7.0f, 6.0f, 5.0f, 4.0f}.shuffle<3, 2, 1, 0>()` returns `Vec_ps{7.0f, 6.0f, 5.0f, 4.0f}` - this is the "identity" shuffle as nothing changes.
- `Vec_ps{7.0f, 6.0f, 5.0f, 4.0f}.shuffle<0, 1, 2, 3>()` returns `Vec_ps{4.0f, 5.0f, 6.0f, 7.0f}`. We have reversed the elements.
- `Vec_pd{7.0, 6.0}.shuffle<1, 1>()` returns `Vec_pd{7.0, 7.0}`, as we choose the highest (1) index as the source for both elements of our new vector.

#### A note on the implementation of shuffling

You might read the C source code for the shuffle implementation, and think that it seems very strange. But, in fact, the x86_64 shuffles should compile to a single CPU instruction, and the AArch64 shuffles should compile to a maximum of 3 CPU instructions, but typically less.

As NEON doesn't have a generic shuffle implementation, instead the code for NEON shuffles was generated by a separate program that searched for every possible shuffle in the shortest number of instructions. As a result it uses a huge switch statement, that then gets optimized out as constant values are passed as an argument. In addition, the SSE2 shuffle implementation also uses a huge switch statement, because the intrinsic requires an immediate value (compile-time constant) and some compilers cannot prove that the function argument then passed to this intrinsic is such.

### Bitcasting between vector types

Any `Vec_` type can be bitcasted to any other `Vec_` type of the same total size. (The elements do not need to be the same size, but the total size of the two vectors must be equal). To do this, you use the `.bitcast<typename To>()` method. Eg `Vec_ps{4.0f}.bitcast<Vec_pi64>()` will re-interpret 4 packed 32-bit floating point values as 2 packed 64-bit signed integers. This particular bitcast is allowed because they are both the same total size of 128 bits.

For 128-bit vectors, bitcasing is usually a no-op (compiles to zero instructions), but the subsequent switch to a different "pipeline" may or may not incur a small performance penalty depending on the hardware. But for the "scalar wrapper" types,  bitcasting is achieved via `memcpy()` which usually gets optimized into a single register move instruction.

You can **not** bitcast a `Compare_` type to any other type, but you can convert between `Compare_` types (see below).

### Conversion between vector types

### Conversion between `Compare_` types

### Type trait and dependent type members