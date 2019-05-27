#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <memory>

#include <cstdint>
#include <cstring>

inline __m128i loadu_si32(const char* src) {
    __m128i r;
    asm ( ".intel_syntax\n"
          "movd  %[RESULT], DWORD PTR [%[SRC]]\n"
          ".att_syntax\n"
          : [RESULT]"=rx"(r)
          : [SRC]"r"(src)
          :              
    );
    return r;
}

inline __m128i loadu_si16(const char* src) {
    __m128i r;
    const std::uint32_t x = (std::uint16_t&)*src;
    asm ( ".intel_syntax\n"
          "movd  %[RESULT], %[SRC]\n"
          ".att_syntax\n"
          : [RESULT]"=rx"(r)
          : [SRC]"r"(x)
          :              
    );
    return r;
}

__m128i load_1_n_6_str(const char* src, std::size_t n) {
    switch ((n+1) / 2) {
    case 3:
        return [&]() {
            const __m128i low = loadu_si32(src);
            const auto high = (std::uint16_t&)*(src + 4);
            return _mm_insert_epi16(low, high, 2);
        }();
    case 2:
        return loadu_si32(src);
    case 1:
        return loadu_si16(src);
    }
}

//------------------------------------------------

static constexpr std::size_t size = 500'000;
auto data = std::make_unique<char[]>(size);

static constexpr std::size_t width = 6;

static void memcpy_load(benchmark::State& state) {
  for (auto _ : state) {
    for (std::size_t i = 0; i < size - width - 1; ++i) {
      __m128i r;
      std::memcpy(&r, &data[i], width);
      benchmark::DoNotOptimize(r);
    }
  }
}
BENCHMARK(memcpy_load);

static void fast_load(benchmark::State& state) {
  for (auto _ : state) {
    for (std::size_t i = 0; i < size - width - 1; ++i) {
      __m128i r = load_1_n_6_str(&data[i], width);
      benchmark::DoNotOptimize(r);
    }
  }
}
BENCHMARK(fast_load);
