#include "HostHash.h"

template<typename Char>
HostSHA256<Char>::HostSHA256(const Char* data, size_t length) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr)
        throw OssException("EVP_MD_CTX_new");

    if (0 == EVP_DigestInit_ex(context, EVP_sha256(), nullptr))
        throw OssException("EVP_DigestInit_ex");

    if (0 == EVP_DigestUpdate(context, data, length * sizeof(char)))
        throw OssException("EVP_DigestUpdate");

    if (0 == EVP_DigestFinal_ex(context, bytes.data(), &bytesLength))
        throw OssException("EVP_DigestFinal_ex");

    EVP_MD_CTX_free(context);
}
template HostSHA256<char>::HostSHA256(const char* data, size_t length);
template HostSHA256<wchar_t>::HostSHA256(const wchar_t* data, size_t length);

template<typename Char>
std::string HostSHA256<Char>::to_string() const {
    std::stringstream ss;
    for (unsigned i = 0; i < bytesLength; ++i)
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
    return ss.str();
}
template std::string HostSHA256<char>::to_string() const;
template std::string HostSHA256<wchar_t>::to_string() const;