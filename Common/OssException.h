#ifndef HASHSELECTION_OSSEXCEPTION_H
#define HASHSELECTION_OSSEXCEPTION_H

#include <iostream>
#include <openssl/err.h>

class OssException final : public std::exception {
    std::string message;
public:
    explicit OssException(std::string message)
    : message(std::move(message) + ": " + ERR_error_string(ERR_get_error(), nullptr)) {}
    const char *what() const noexcept override { return message.c_str(); }
};

#endif //HASHSELECTION_OSSEXCEPTION_H
