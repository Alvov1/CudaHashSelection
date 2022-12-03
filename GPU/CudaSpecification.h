#ifndef HASHSELECTION_CUDASPECIFICATION_H
#define HASHSELECTION_CUDASPECIFICATION_H

#include "Console.h"
#include "CudaException.h"

namespace Hardware {
    void checkCudaDevices() {
        int nDevices = 0;
        auto code = cudaGetDeviceCount(&nDevices);
        if(code != cudaSuccess)
            throw CudaException(code);

        if(nDevices == 0)
            throw std::runtime_error("Not found any cuda-capable devices available.");

        Console::cout << "Found " << nDevices << " cuda capable devices:" << Console::endl;
        for(int i = 0; i < nDevices; ++i) {
            cudaDeviceProp device {};
            code = cudaGetDeviceProperties(&device, i);
            if(code != cudaSuccess)
                throw CudaException(code);

            Console::cout << i + 1 << ". " << std::string(device.name) << ", KHz: " << device.memoryClockRate
            << ", bus width: " << device.memoryBusWidth << "." << Console::endl;

        }
    }
}

#endif //HASHSELECTION_CUDASPECIFICATION_H
