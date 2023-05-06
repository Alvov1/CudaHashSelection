#include "HostHash.h"

class OssException final : public std::exception {
    std::string message;
public:
    explicit OssException(std::string message)
            : message(std::move(message) + ": " + ERR_error_string(ERR_get_error(), nullptr)) {}
    [[nodiscard]] const char *what() const noexcept override { return message.c_str(); }
};

template<typename Char>
HostSHA256::HostSHA256(const Char* data, size_t length) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr)
        throw OssException("EVP_MD_CTX_new");

    if (0 == EVP_DigestInit_ex(context, EVP_sha256(), nullptr))
        throw OssException("EVP_DigestInit_ex");

    if (0 == EVP_DigestUpdate(context, data, length * sizeof(char)))
        throw OssException("EVP_DigestUpdate");

    if (0 == EVP_DigestFinal_ex(context, bytes.data(), nullptr))
        throw OssException("EVP_DigestFinal_ex");

    EVP_MD_CTX_free(context);
}
template HostSHA256::HostSHA256(const char* data, size_t length);
template HostSHA256::HostSHA256(const wchar_t* data, size_t length);

std::string HostSHA256::to_string() const {
    std::stringstream ss;
    for (unsigned i = 0; i < Sha256DigestLength; ++i)
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
    return ss.str();
}

const std::array<unsigned char, HostSHA256::Sha256DigestLength>& HostSHA256::get() const {
    return bytes;
}
