#ifndef HASHSELECTION_DEVICEHASH_H
#define HASHSELECTION_DEVICEHASH_H

#define DEVICE __device__

#include <cstdio>
#include <cstdint>
#include "Array.h"

class DeviceSHA256 {
    static constexpr size_t BlockSize = (512 / 8);
    static constexpr size_t DigestSize = ( 256 / 8);
    using DigestType = Array<uint32_t>;

    DEVICE void update_(unsigned char const *message, size_t len);
    DEVICE void transform_(unsigned char const *message, unsigned int block_nb);

    DEVICE static inline uint32_t SHA2_SHFR(uint32_t x, uint32_t n) { return x >> n; }
    DEVICE static inline uint32_t SHA2_ROTR(uint32_t x, uint32_t n) { return ((x >> n) | (x << ((sizeof(x) << 3) - n))); }
    DEVICE static inline uint32_t SHA2_ROTL(uint32_t x, uint32_t n) { return ((x << n) | (x >> ((sizeof(x) << 3) - n))); }
    DEVICE static inline uint32_t SHA2_CH(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) ^ (~x & z)); }
    DEVICE static inline uint32_t SHA2_MAJ(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) ^ (x & z) ^ (y & z)); }
    DEVICE static inline uint32_t SHA256_F1(uint32_t x) { return (SHA2_ROTR(x, 2) ^ SHA2_ROTR(x, 13) ^ SHA2_ROTR(x, 22)); }
    DEVICE static inline uint32_t SHA256_F2(uint32_t x) { return (SHA2_ROTR(x, 6) ^ SHA2_ROTR(x, 11) ^ SHA2_ROTR(x, 25)); }
    DEVICE static inline uint32_t SHA256_F3(uint32_t x) { return (SHA2_ROTR(x, 7) ^ SHA2_ROTR(x, 18) ^ SHA2_SHFR(x, 3)); }
    DEVICE static inline uint32_t SHA256_F4(uint32_t x) { return (SHA2_ROTR(x, 17) ^ SHA2_ROTR(x, 19) ^ SHA2_SHFR(x, 10)); }
    DEVICE static inline uint32_t SHA2_PACK32(unsigned char const *str) {
        return ((uint32_t) *((str) + 3)) | ((uint32_t) *((str) + 2) << 8)
               | ((uint32_t) *((str) + 1) << 16) | ((uint32_t) *((str) + 0) << 24);
    }
    DEVICE static inline void SHA2_UNPACK32(uint32_t x, unsigned char *str) {
        *((str) + 3) = (uint8_t) ((x)); *((str) + 2) = (uint8_t) ((x) >> 8);
        *((str) + 1) = (uint8_t) ((x) >> 16); *((str) + 0) = (uint8_t) ((x) >> 24);
    }

    DEVICE static constexpr unsigned sha256_k(size_t index);

    unsigned int tot_len_{};
    unsigned int len_{};
    unsigned char block_[2 * BlockSize]{};
    uint32_t digest_[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

    DEVICE DigestType finalDigest();
public:
    DEVICE inline DeviceSHA256(const char* value, size_t size) {
        auto const *in_uchar = reinterpret_cast<unsigned char const *>(value);
        update_(in_uchar, size);
    }
    DEVICE DeviceSHA256(DeviceSHA256 const &) = delete;
    DEVICE DeviceSHA256 &operator=(DeviceSHA256 const &) = delete;

    DEVICE Array<char> count();
};

#endif //HASHSELECTION_DEVICEHASH_H
