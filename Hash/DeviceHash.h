#ifndef HASHSELECTION_DEVICEHASH_H
#define HASHSELECTION_DEVICEHASH_H

#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>

using u64 = uint64_t;
using u32 = uint32_t;
using u8 = uint8_t;

class DeviceSHA256 final {
    std::array<unsigned char, 32> data {};
    void transform(std::array<u32, 8>& state);
public:
    template <typename Char = char>
    DeviceSHA256(const Char* input, std::size_t length);
    const std::array<unsigned char, 32>& get() const;
    [[nodiscard]] std::string to_string() const;
};

#endif //HASHSELECTION_DEVICEHASH_H
