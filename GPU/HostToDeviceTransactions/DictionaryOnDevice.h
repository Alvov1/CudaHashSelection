#ifndef HASHSELECTION_DICTIONARYONDEVICE_H
#define HASHSELECTION_DICTIONARYONDEVICE_H

#include <iostream>
#include <string>
#include <vector>

#include "Timer.h"
#include "CudaException.h"

class DictionaryOnDevice final {
    std::vector<char*> hostPointersArray;
    size_t hostPointersArraySize = 0;
    char** devicePointersArray = nullptr;
public:
    DictionaryOnDevice(const std::vector<std::string>& words);

    char** getArray() const { return devicePointersArray; }
    size_t size() const { return hostPointersArraySize; }

    ~DictionaryOnDevice();
};

#endif //HASHSELECTION_DICTIONARYONDEVICE_H
