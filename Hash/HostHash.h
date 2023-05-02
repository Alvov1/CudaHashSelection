#ifndef HASHSELECTION_HOSTHASH_H
#define HASHSELECTION_HOSTHASH_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <openssl/evp.h>
#include <openssl/err.h>

class OssException;

template <typename Char = char>
class HostSHA256 final {
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes {};
    unsigned bytesLength {};
public:
    HostSHA256(const Char* data, std::size_t length);
    [[nodiscard]] std::string to_string() const;
};

class OssException final : public std::exception {
    std::string message;
public:
    explicit OssException(std::string message)
    : message(std::move(message) + ": " + ERR_error_string(ERR_get_error(), nullptr)) {}
    [[nodiscard]] const char *what() const noexcept override { return message.c_str(); }
};

#endif //HASHSELECTION_HOSTHASH_H
