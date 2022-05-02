#ifndef HASHSELECTION_HOSTHASH_H
#define HASHSELECTION_HOSTHASH_H

#include <iostream>
#include <array>
#include <sstream>
#include <iomanip>

class HostSHA256 {
    static constexpr size_t BlockSize = (512 / 8);
    static constexpr size_t DigestSize = ( 256 / 8);
    using DigestType = std::array<uint32_t, 8>;

    void update_(unsigned char const *message, size_t len);
    void transform_(unsigned char const *message, unsigned int block_nb);

    static inline uint32_t SHA2_SHFR(uint32_t x, uint32_t n) { return x >> n; }
    static inline uint32_t SHA2_ROTR(uint32_t x, uint32_t n) { return ((x >> n) | (x << ((sizeof(x) << 3) - n))); }
    static inline uint32_t SHA2_ROTL(uint32_t x, uint32_t n) { return ((x << n) | (x >> ((sizeof(x) << 3) - n))); }
    static inline uint32_t SHA2_CH(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) ^ (~x & z)); }
    static inline uint32_t SHA2_MAJ(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) ^ (x & z) ^ (y & z)); }
    static inline uint32_t SHA256_F1(uint32_t x) { return (SHA2_ROTR(x, 2) ^ SHA2_ROTR(x, 13) ^ SHA2_ROTR(x, 22)); }
    static inline uint32_t SHA256_F2(uint32_t x) { return (SHA2_ROTR(x, 6) ^ SHA2_ROTR(x, 11) ^ SHA2_ROTR(x, 25)); }
    static inline uint32_t SHA256_F3(uint32_t x) { return (SHA2_ROTR(x, 7) ^ SHA2_ROTR(x, 18) ^ SHA2_SHFR(x, 3)); }
    static inline uint32_t SHA256_F4(uint32_t x) { return (SHA2_ROTR(x, 17) ^ SHA2_ROTR(x, 19) ^ SHA2_SHFR(x, 10)); }
    static inline uint32_t SHA2_PACK32(unsigned char const *str) {
        return ((uint32_t) *((str) + 3)) | ((uint32_t) *((str) + 2) << 8)
               | ((uint32_t) *((str) + 1) << 16) | ((uint32_t) *((str) + 0) << 24);
    }
    static inline void SHA2_UNPACK32(uint32_t x, unsigned char *str) {
        *((str) + 3) = (uint8_t) ((x)); *((str) + 2) = (uint8_t) ((x) >> 8);
        *((str) + 1) = (uint8_t) ((x) >> 16); *((str) + 0) = (uint8_t) ((x) >> 24);
    }

    const static uint32_t sha256_k[];
    unsigned int tot_len_{};
    unsigned int len_{};
    unsigned char block_[2 * BlockSize]{};
    DigestType digest_{
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    DigestType finalDigest();
public:
    inline HostSHA256(const char* value, size_t size) {
        auto const *in_uchar = reinterpret_cast<unsigned char const *>(value);
        update_(in_uchar, size);
    }
    explicit inline HostSHA256(const std::string& value) : HostSHA256(value.c_str(), value.size()) {}
    HostSHA256(HostSHA256 const &) = delete;
    HostSHA256 &operator=(HostSHA256 const &) = delete;

    std::string out();
};

inline std::ostream& operator<<(std::ostream& os, HostSHA256& value) {
    os << value.out(); return os;
}

#endif //HASHSELECTION_HOSTHASH_H
