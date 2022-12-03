#include "DictionaryOnDevice.h"

DictionaryOnDevice::DictionaryOnDevice(const std::vector<std::string> &words)
        : hostPointersArraySize(words.size()),
          hostPointersArray(std::vector<char*>(words.size() * sizeof(char*))) {
    for(unsigned i = 0; i < hostPointersArraySize; ++i) {
        const auto& current = words[i];
        auto code = cudaMalloc(&hostPointersArray[i], current.size() * sizeof(char));
        if(code != cudaSuccess)
            throw CudaException(code);
        code = cudaMemcpy(hostPointersArray[i], current.c_str(), current.size() * sizeof(char), cudaMemcpyHostToDevice);
        if(code != cudaSuccess)
            throw CudaException(code);
    }

    size_t arrayMemorySpace = hostPointersArraySize * sizeof(char*);
    auto code = cudaMalloc(&devicePointersArray, arrayMemorySpace);
    if(code != cudaSuccess)
        throw CudaException(code);
    code = cudaMemcpy(devicePointersArray, hostPointersArray.data(), arrayMemorySpace, cudaMemcpyHostToDevice);
    if(code != cudaSuccess)
        throw CudaException(code);
}

DictionaryOnDevice::~DictionaryOnDevice() {
    for(unsigned i = 0; i < hostPointersArraySize; ++i) {
        auto code = cudaFree(hostPointersArray[i]);
        if (code != cudaSuccess)
            std::cerr << CudaException(code).what() << std::endl;
    }
    auto code = cudaFree(devicePointersArray);
    if(code != cudaSuccess)
        std::cerr << CudaException(code).what() << std::endl;
}
