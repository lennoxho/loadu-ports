#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include <cstdint>
#include <cstdlib>

#ifdef _MSC_VER
#include <intrin.h>
#endif

inline __m128i loadu_si32(const char* src) {
#ifdef _MSC_VER
    __m128i r = _mm_setzero_si128();
    r = _mm_insert_epi32(r, (std::uint32_t&)*src, 0);
#elif defined(__INTEL_COMPILER) || defined(__clang__)
    __m128i r = _mm_loadu_si32(src);
#else
    // Optimal
    __m128i r;
    asm ( "movd  %[RESULT], DWORD PTR [%[SRC]]"
          : [RESULT]"=rx"(r)
          : [SRC]"r"(src)
          :              
    );
#endif
    return r;
}

inline __m128i loadu_si16(const char* src) {
#ifdef _MSC_VER
    __m128i r = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, (std::uint16_t&)*src);
    // MSVC somehow decides to generate one extra movzx instruction when we manually insert byte?
    //__m128i r = _mm_setzero_si128();
    //r = _mm_insert_epi16(r, (std::uint16_t&)*src, 0);
#elif defined(__INTEL_COMPILER) || defined(__clang__)
    __m128i r = _mm_loadu_si16(src);
#else
    // Optimal
    __m128i r;
    const std::uint32_t x = (std::uint16_t&)*src;
    asm ( "movd  %[RESULT], %[SRC]"
          : [RESULT]"=rx"(r)
          : [SRC]"r"(x)
          :              
    );
#endif
    return r;
}

inline __m128i loadu_si48(const char* src) {
    const __m128i low = loadu_si32(src);
    const auto high = (std::uint16_t&)*(src + 4);
    return _mm_insert_epi16(low, high, 2);
}

__m128i load_1_n_6_str(const char* src, std::size_t n) {
    static constexpr decltype(loadu_si32)* jmp_table[] = { loadu_si16, loadu_si32, loadu_si48 };
    return jmp_table[(n-1)/2](src);
}
