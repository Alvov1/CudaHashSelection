#ifndef HASHSELECTION_ARRAYONDEVICE_H
#define HASHSELECTION_ARRAYONDEVICE_H

#include "CudaException.h"

template <typename Base = char>
class ArrayOnDevice final {
    Base* pointer = nullptr;
    size_t length = 0;
public:
    ArrayOnDevice() = default;
    ArrayOnDevice(size_t withSize);
    ArrayOnDevice(const Base* data, size_t length);

    Base* get() const { return pointer; }
    size_t size() const { return length; }
    std::basic_string<Base> readBack() const;

    ~ArrayOnDevice();

    ArrayOnDevice(const ArrayOnDevice& copy) = delete;
    ArrayOnDevice& operator=(const ArrayOnDevice& assign) = delete;
    ArrayOnDevice(ArrayOnDevice&& move) noexcept = default;
    ArrayOnDevice& operator=(ArrayOnDevice&& moveAssign) noexcept = delete;
};

template<typename Base>
ArrayOnDevice<Base>::ArrayOnDevice(size_t withSize): length(withSize) {
    auto code = cudaMalloc(&pointer, sizeof(Base) * (withSize + 1));
    if(code != cudaSuccess)
        throw CudaException(code);
    code = cudaMemset(pointer, 0, withSize + 1);
    if(code != cudaSuccess)
        throw CudaException(code);
}

template<typename Base>
ArrayOnDevice<Base>::ArrayOnDevice(const Base *data, size_t length): ArrayOnDevice(length) {
    auto code = cudaMemcpy(pointer, data, sizeof(Base) * length, cudaMemcpyHostToDevice);
    if(code != cudaSuccess)
        throw CudaException(code);
}

template<typename Base>
std::basic_string<Base> ArrayOnDevice<Base>::readBack() const {
    std::basic_string<Base> result(length + 1, Base());
    auto code = cudaMemcpy(result.data(), pointer, sizeof(Base) * length, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess)
        throw CudaException(code);
    return result;
}

template<typename Base>
ArrayOnDevice<Base>::~ArrayOnDevice() {
    auto code = cudaFree(pointer);
    if(code != cudaSuccess)
        std::cerr << CudaException(code).what() << std::endl;
}

#endif //HASHSELECTION_ARRAYONDEVICE_H
