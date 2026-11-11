/* Exact binary rank. Rows are packed in little-endian 64-bit words.
 * The buffer is modified in place. No allocation or external dependencies. */
#include <stddef.h>
#include <stdint.h>

size_t gf2_rank(uint64_t *a, size_t rows, size_t bits) {
    const size_t words = (bits + 63) / 64;
    size_t rank = 0;
    for (size_t col = bits; col-- > 0 && rank < rows;) {
        const size_t word = col / 64;
        const uint64_t mask = UINT64_C(1) << (col % 64);
        size_t pivot = rank;
        while (pivot < rows && !(a[pivot * words + word] & mask)) ++pivot;
        if (pivot == rows) continue;
        if (pivot != rank) {
            for (size_t k = 0; k <= word; ++k) {
                uint64_t tmp = a[rank * words + k];
                a[rank * words + k] = a[pivot * words + k];
                a[pivot * words + k] = tmp;
            }
        }
        for (size_t i = rank + 1; i < rows; ++i) {
            if (a[i * words + word] & mask) {
                for (size_t k = 0; k <= word; ++k)
                    a[i * words + k] ^= a[rank * words + k];
            }
        }
        ++rank;
    }
    return rank;
}
