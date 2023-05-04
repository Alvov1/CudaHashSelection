#ifndef HASHSELECTION_HOSTHASH_H
#define HASHSELECTION_HOSTHASH_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <openssl/evp.h>
#include <openssl/err.h>

class HostSHA256 final {
    std::array<unsigned char, 32> bytes {};
public:
    template <typename Char = char>
    HostSHA256(const Char* data, std::size_t length);
    const std::array<unsigned char, 32>& get() const;
    [[nodiscard]] std::string to_string() const;
};

#endif //HASHSELECTION_HOSTHASH_H
