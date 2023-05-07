#ifndef HASHSELECTION_HOSTHASH_H
#define HASHSELECTION_HOSTHASH_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <openssl/evp.h>
#include <openssl/err.h>

namespace Hash {
    class HostSHA256 final {
        static constexpr auto Sha256DigestLength = 32;
        std::array<unsigned char, Sha256DigestLength> bytes{};
    public:
        template<typename Char = char>
        HostSHA256(const Char *data, std::size_t length);
        const std::array<unsigned char, Sha256DigestLength> &get() const;
        [[nodiscard]] std::string to_string() const;
    };
}

#endif //HASHSELECTION_HOSTHASH_H
