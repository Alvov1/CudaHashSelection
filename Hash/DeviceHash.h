#ifndef HASHSELECTION_DEVICEHASH_H
#define HASHSELECTION_DEVICEHASH_H

#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>

#define DEVICE __device__

using u64 = uint64_t;
using u32 = uint32_t;
using u8 = uint8_t;

class DeviceSHA256 final {
    unsigned char data[32] {};
    DEVICE void transform(u32* state);
public:
    template <typename Char = char>
    DEVICE DeviceSHA256(const Char* input, std::size_t length);
    DEVICE const unsigned char* get() const;
};

#endif //HASHSELECTION_DEVICEHASH_H
