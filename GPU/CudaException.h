#ifndef HASHSELECTION_CUDAEXCEPTION_H
#define HASHSELECTION_CUDAEXCEPTION_H

#include <exception>

class CudaException final: public std::exception {
    std::string message;
public:
    CudaException(cudaError_t code)
    : message(std::string("Cuda exception: ") + cudaGetErrorString(code)) {}
    const char* what() const noexcept override { return message.c_str(); }
};

#endif //HASHSELECTION_CUDAEXCEPTION_H
