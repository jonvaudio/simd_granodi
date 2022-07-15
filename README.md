# simd_granodi.h

### Easy C/C++ SIMD in a single header file, supporting AArch64 NEON and x86_64 SSE2

## Features
- Supports C99, and C++11 with extensive operator overloading, type traits, templated conversions etc.
- Full test coverage.
- Easy and type-safe comparison and branchless conditional selection of results.
- The C++ classes are written in terms of the C implementation.
- The C++ classes are (mostly) immutable, apart from the `+=`, `-=`, `*=`, `/=`, `&=`, `|=`, `^=` operators. Most non-static methods are marked `const` and return a new value.
- Templated methods and type traits make it easy to write C++ SIMD code that works on several different types.
- The C implementation mostly uses macros (where arguments are evaluated no more than once), for faster debug builds. C code using the library should take almost the same amount of time to compile as if you had used the intrinsics directly.
- The included generic, non-vector implementation allows for compilation on any target. (This implementation can be forced by defining `SIMD_GRANODI_FORCE_GENERIC`). It also serves as documentation for those not familiar with SIMD intrinsics.
- Avoids undefined behaviour.
- Tested on GCC, Clang, and MSVC++.
- `simd_granodi.h` is the only file you need.
- Designed to be easy enough to use for someone who has never done any SIMD programming before.

### Why 64-bit only? Why only SSE2?

This library was written so that the author could easily write cross-platform Audio DSP code that runs on all x86_64 machines, and newer AArch64 machines (ie the latest hardware from Apple). It targets 128-bit SIMD types (32x4 or 64x2). However, the generic implementation should be able to target any hardware.

The x86\_64 implementation limits itself to SSE2, as many otherwise-capable modern low-end CPUs do not support AVX, SSE2 is guaranteed to be implemented on all x86_64 machines, and later SSE instructions (eg SSE 4.2) only provide a marginal speed improvement to the functionality of this library. Also, NEON only supports 128-bit vectors. However, it would be very easy to add AVX support in future, and simply wrap 2 or 4 128-bit registers on NEON.

### Generic "fallback" implementation

If the header cannot detect that you are on x64 or AArch64 using one of Clang, GCC, or MSVC++, it will revert to using a "generic" implementation which uses standard C/C++ scalar operations. This means that there is no risk of writing SIMD code, and then suddenly discovering you need to compile for a target that has no SIMD hardware.

### Slow / non-vector functions

Some platforms do not have intrinsic functions for some SIMD operations, and so they are implemented generically and may be slower. A list of these functions/macros, per-platform, is contained in a comment at the start of the `simd_granodi.h` file. If you are using the C++ classes, you may wish to search for those names in the file to see which methods they correspond to.

## C++ documentation
### Namespaces

All of the C++ code is inside the namespace `simd_granodi`. All of the code examples below assume you are `using namespace simd_granodi;`, but you may choose to do something like `namespace sg = simd_granodi;`.

The C functions / macros are **not** inside a namespace (because of the use of macros), but typically have the prefix `sg_`.

### Vector types
All of the following vector types are 128-bit in size:

- `Vec_pi32` - Vector of 4 **32**-bit signed **p**acked **i**ntegers. AKA `Vec_s32x4`
- `Vec_pi64` - Vector of 2 **64**-bit signed **p**acked **i**ntegers. AKA `Vec_s64x2`
- `Vec_ps` - Vector of 4 **p**acked **s**ingle-precision `float`. AKA `Vec_f32x4`
- `Vec_pd` - Vector of 2 **p**acked **d**ouble-precision `float`. AKA `Vec_f64x2`

The following vector types are 64-bit in size:

- `Vec_s32x2` - Vector of 2 `int32_t`
- `Vec_f32x2` - Vector of 2 `float`

**Note:** On SSE2, `Vec_s32x2` and `Vec_f32x2` are implemented generically. See below for explanation. 

The following are "scalar wrapper" types, which allow you to write templated code that operates either on built-in C++ types or SIMD vectors:

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
- `Compare_s32x2` - Result of comparing two `Vec_s32x2`
- `Compare_f32x2` - Result of comparing two `Vec_f32x2`

The following are type-safe comparison types for the equivalent "scalar wrapper" types, allowing you to write templated code that operates either on vectors or C++ built-in types. They are a simple wrapper for `bool`, and should get optimized out of existence when used:

- `Compare_s32x1` - Result of comparing two `Vec_s32x1`
- `Compare_s64x1` - Result of comparing two `Vec_s64x1`
- `Compare_f32x1` - Result of comparing two `Vec_f32x1`. AKA `Compare_ss`
- `Compare_s64x1` - Result of comparing two `Vec_f64x1`. AKA `Compare_sd`

### A note on `Vec_s32x2` and `Vec_f32x2`

On NEON, `Vec_s32x2` and `Vec_f32x2` are both native types. But on SSE2, they are emulated via a struct containing two `int32_t` or two `float` respectively. These types are useful as they take up less space than `Vec_pi32` and `Vec_ps` (for example, if you hold a large array of them). But for long running calculations, you can use the following type alias to convert to the fastest in-register type with size of at least 2 elements::

- `Vec_s32x2::fast_register_t` - defined as `Vec_pi32` on SSE2, and `Vec_s32x2` on all other platforms
- `Vec_f32x2::fast_register_t` - defined as `Vec_ps` on SSE2, and `Vec_f32x2` on all other platforms.

These are type aliases, so in order to use them you must use the `.to<NewType>()` templated method to convert. For example, `Vec_f32x2{5.0f, 4.0f}.to<typename Vec_f32x2::fast_register_t>()` will give you the fastest in-register vector for your platform with those values.

### A note on using MSVC++ on x64

In order to obtain good performance when compiling x64 code using the SIMD C++ classes, it is recommended to take the following steps:

##### Use the `sg_vectorcall` function macro

Use the `sg_vectorcall(f)` macro to define your own functions which take `float`, `double`, or any SIMD type or SIMD class wrapper type as an argument, to avoid unnecessary loads and stores. On MSVC++ under x64, this macro is defined as:

```cpp
#define sg_vectorcall(f) __vectorcall f
````

and on all other platforms, this macro is defined as the identity macro:

```cpp
#define sg_vectorcall(f) f
```

and so is harmless.

Examples of use:

```cpp
float sg_vectorcall(my_func)(const float x) {
    return x + 12.0f;
}

Vec_ps sg_vectorcall(my_func_simd)(const Vec_ps x) {
	return x + 12.0f;
}
```

##### Pass SIMD C++ class types by value, not by `const` reference

On MSVC++, passing SIMD class types by `const` reference can introduce unnecessary loads and stores.

##### Compile with the `/GS-` (the important part of that flag being the `-`)

Functions which take an argument the same type as the C++ SIMD classes cause MSVC++ to place a security cookie on the stack before that function is called, and check that cookie again when the function returns. (This only happens if the function is **not** inlined). This is a sensible way to check for stack corruption, but can slow things down if you repeatedly call a function which (for example) takes a `Vec_ps` as an argument, but is large enough to not get inlined.

### Constructors

#### Default constructor

- All vector types are default-constructed to hold a value of zero. This is for safety and convenience, and this assignment typically gets optimized out.
- All comparison types are default-constructed to hold a value of `false`. With vector comparisons, this is a bit-mask comprised of all zeros. With "scalar wrapper" comparisons, this is a `bool` with value `false`.

#### "Broadcast" constructors

- All vector types have a "broadcast" constructor that takes a single value and "broadcasts" it to all elements of the vector. For example, `Vec_ps{3.0f}` will result in the vector `Vec_ps{3.0f, 3.0f, 3.0f, 3.0f}`. This also allows for convenient arithmetic with constants, for example `Vec_ps{3.0f} + 1.0f` will give the vector `{4.0f, 4.0f, 4.0f, 4.0f}` as the `1.0f` is implicitly constructed into a `Vec_ps`.
- Note that vector types do **not** have a broadcast constructor that takes an equivalent "scalar wrapper" type as an argument, due to constructor overload ambiguity when using literals/constants in code. Ie you can **not** do `Vec_ps{Vec_f32x1{1.0f}}`. However there are easy workarounds for this: construct from the `.data()` element of the scalar wrapper type, or convert the scalar wrapper type using the `.to<NewType>()` method.
- All comparison types also have a "broadcast" constructor which accepts a `bool`. For vector comparisons, **all** of the bits of the bit-mask are set to `0` if this is `false`, or `1` if this is true. For the "scalar wrapper" types, this sets the value of the `bool` member.

The rationale behind the broadcast constructors, is that they allow constants to be easily mixed in with vector code. For example, `Vec_pd{5.0, 4.0} + 2.0` will give the vector `{7.0, 6.0}` as `2.0` is implicitly constructed into a `Vec_pd` of `{2.0, 2.0}`.

There is no `Vec_pd` constructor that accepts a single `Vec_f64x1` scalar wrapper type to broadcast to all elements. This is because `Vec_pd{5.0, 4.0} + 2.0` would not compile, as the compiler would not know whether to interpret it as `Vec_pd{5.0, 4.0f} + Vec_pd{Vec_f64x1{2.0}}` or `Vec_pd{5.0, 4.0f} + Vec_pd{2.0}`, as both `Vec_pd` and `Vec_f64x1` can be constructed from a `double`.

#### Element-wise constructors

- All Vector types can be constructed by specifying the value of each element. Following the SSE2 convention, the elements are specified in reverse order. For example, `Vec_ps{3.0f, 2.0f, 1.0f, 0.0f}` creates a vector with the value 3 at index 3, value 2 at index 2, value 1 at index 1, and value 0 at index 0. When these values are compile-time constants, this constructor is typically optimized into a single instruction, otherwise it may take several instructions.
- Vector types with 4 elements can be constructed by specifying the value of the lowest 2 or 3 elements, and the upper 1 or 2 elements will be zeroed. Eg `Vec_ps{3.0f, 7.0f}` evaluates as `{0.0f, 0.0f, 3.0f, 7.0f}`.
- All comparison types also have a "vector" constructor that takes a `bool` to specify a bit-mask for each element. `true` will be interpreted as **all** bits set to 1, and `false` will be interpreted as all bits set to 0. The reverse ordering convention is the same as for vector types.

### Load and store vectors from/to pointers to element type

Every `Vec_` type has `load`, `loadu`, `store`, and `storeu` methods. **Warning:** The `load` and `store` methods take pointers to data that **must be correctly aligned**.

The `load` and `loadu` (`u` means unaligned) are `static` methods that take a pointer to the vector's element type (ie, `int32_t*`, `int64_t*`, `float*`, or `double*`, depending on the vector type) and construct a new vector from the elements pointed to. For example, `auto vec = Vec_pd::loadu(&my_double_array[4])` will result in a vector of value `{my_double_array[5], my_double_array[4]}`.

The `store` and `storeu` methods are similar, except they are **not** `static` and return `void`. They store the vector at the given element pointer location. Eg using the `vec` variable from the previous paragraph, you could then do `vec.store(&my_double_array[4])` to store the vector back to where you loaded it from.

### Vector element access

The elements of a vector can be accessed with the templated `.get<int32_t>()` method. The index is passed as the template argument. An out of range index will cause a compile time error. Example: `Vec_pd{4.0, 3.0}.get<1>()` will return a `double` with value `4.0`.

Every vector and comparison type also has a `.data()` method that allows access to the underlying, built-in representation of that type (eg a value of type `float` or `__m128d`).

### Setting vector elements

An element of a vector can be changed with the templated `.set<int32_t>(new_val)` method. But note that **this method is const, and returns a new vector, leaving the original vector unchanged**.

Eg `Vec_ps{7.0f, 2.0f, 5.0f, 4.0f}.set<2>(6.0f)` returns a new `Vec_ps` of value `{7.0f, 6.0f, 5.0f, 4.0f}`. Note that this is efficiently implemented in-register, and on most good compilers will **not** result in any loads or stores.

### Arithmetic operators

All `Vec_` types implement the following standard arithmetic operators: `+=`, `+`, `-=`, `-`, `*=`, `*`, `/=`, `/`. Also, integer types support both the pre- and postfix `++` and `--` operators.

### Bitwise operators

All `Vec_` types, including floating-point types, implement the following bitwise operators: `&=`, `&`, `|=`, `|`, `^=`, `^`, `~`.

**Note:** You might assume that, since `Compare_` types use bitwise operations internally to mask or "select" a result, that they also implement these bitwise operators. However, for reasons of type safety, they do not implement these operators at all. Instead, they only implement logical operators.

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

### Bitcasting between `Vec_` types

Any `Vec_` type can be bitcasted to any other `Vec_` type of the same total size. (The elements do not need to be the same size, but the total size of the two vectors must be equal). To do this, you use the `.bitcast<typename To>()` method. Eg `Vec_ps{4.0f}.bitcast<Vec_pi64>()` will re-interpret 4 packed 32-bit floating point values as 2 packed 64-bit signed integers. This particular bitcast is allowed because they are both the same total size of 128 bits.

For 128-bit vectors, bitcasing is usually a no-op (compiles to zero instructions), but the subsequent switch to a different "pipeline" may or may not incur a small performance penalty depending on the hardware. But for the "scalar wrapper" types,  bitcasting is achieved via `memcpy()` which usually gets optimized into a single register move instruction.

You can **not** bitcast a `Compare_` type to any other type (including another `Compare_` type), but you can convert between `Compare_` types (see below).

### Conversion between `Vec_` types

You can convert to and from any `Vec_` types. In general, this is achieved using the templated `.to<typename To>()` method. **However**, this method is **not** implemented for converting **from** a floating point type **to** an integer type. This is because a rounding method needs to be specified, using one of the following templated methods: `.truncate<typename To>()`, `.floor<typename To>()`, or `.nearest<typename To>()`.

#### Converting from 32x4 vectors to 64x2 vectors

When you convert from a 32x4 vector (i.e. Vec_pi32 and Vec_ps) to a 64x2 vector (i.e. Vec_pi64 and Vec_pd), the lowest two elements from the 32x4 vector (at indexes 0 and 1) will be converted to new values for the 64x2 vector (and placed at indexes 0 and 1), and the highest two elements from the 32x4 vector (at indexes 2 and 3) will be discarded.

#### Converting from 64x2 vectors to 32x4 vectors

When you convert from a 64x2 vector (i.e. Vec_pi64 and Vec_pd) to a 32x4 vector (i.e. Vec_pi32 and Vec_ps), the elements from the 64x2 vector (at indexes 0 and 1) will be converted to new values and placed into indexes 0 and 1 of the the 32x4 vector. Indexes 2 and 3 of the 32x4 vector will be set to zero.

#### Converting to and from "scalar wrapper" vector types

- A "scalar wrapper" vector can be converted to any other vector type using the conversion methods.
- A vector with more than one element **cannot** be converted to a "scalar wrapper" vector type using the conversion methods. Instead, you must use the templated `.get<int32_t index>()` method to choose an element from the vector, then use that element to construct a new "scalar wrapper" vector type.

#### Converting between float types

- When converting a 64-bit float type to a 32-bit float type, the platform's default rounding method will be used. This is usually "round to nearest", with ties rounding to even.
- When converting a 32-bit float vector type to a 64-bit float vector type, there will be no loss of precision as the 64-bit type can represent the 32-bit type exactly.

#### Converting from float types to integer types

When converting from a float type to an integer type, you **cannot** use the templated `.to<typename To>()` method. Instead, you must use one of the following methods:

- `.truncate<typename To>()`: Round towards zero.
- `.floor<typename To>()`: Round towards minus infinity.
- `.nearest<typename To>()`: Round to nearest, with ties rounding to even.

#### The static `::from(const FromType v)` method...

... allows you to construct a new `Vec_` type from a different type, with the exact same behaviour as the `.to<typename To>()` method. As with `.to<typename To>()`, you cannot construct an integer type from a float type.

### Conversion between `Compare_` types

The `.to<typename To>()` method also works for `Compare_` types, to resize bitmasks. (For example, comparing two `Vec_pd` and using the result of that comparison to select `Vec_pi64` results).

For `Compare_` types whose elements are of different sizes (i.e. 32x4 or 64x2), the conversion behaviour is identical to that described for vectors above.

### Type traits and member type aliases

To aid in templated programming, all `Vec_` types define the following type aliases:

- `elem_t`: The built-in type that corresponds to the elements of the vector. I.e. `Vec_pi32:elem_t` is `int32_t`.
- `compare_t`: The `Compare_` type that corresponds to the vector. I.e. `Vec_pi32::compare_t` is `Compare_pi32`.
- `fast_register_t`: The fastest in-register type that has at least as many number of elements. For example, on SSE2, `Vec_f32x2::fast_register_t` is defined as `Vec_ps`, but on all other platforms it is defined as `Vec_f32x2`.

All `Vec_` types also define the following `static constexpr` members:

- `is_int_t` - a `bool` indicating whether the vector is an integer type or not
- `is_float_t` - a `bool` indicating whether the vector is a floating point type or not
- `elem_size` - a `std::size_t` giving the size, in bytes, of each element of the vector. Eg `Vec_ps::elem_size` is equal to 4, because a `float` takes up 4 bytes.
- `elem_count` - a `std::size_t` giving the number of elements the vector has. Eg `Vec_pd::elem_count` is 2, because it contains two values of type `double`.

All floating point `Vec_` types define the following type alias:

- `fast_convert_int_t` - An integer type that it is fast to convert to and from. For example, on SSE2, `Vec_pd::fast_convert_int_t` is defined as `Vec_pi32`, but on NEON, it is `Vec_pi64`.

### Type "finder" classes

- `SGType<typename ElemType, std::size_t ElemCount>` - This `struct` allows you to query its `value` member to find a vector type with the given element type and number of elements.
- `SGIntType<std::size_t ElemSize, std::size_t ElemCount>` - allows you to find an integer vector with the given element size (in bytes) and number of elements.
- `SGFloatType<Std::size_t ElemSize, std::size_t ElemCoun>` - as with `SGIntType`, but with floating point types.

### Utility and convenience methods