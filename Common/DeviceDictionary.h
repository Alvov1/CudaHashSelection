#ifndef HASHSELECTION_DEVICEDICTIONARY_H
#define HASHSELECTION_DEVICEDICTIONARY_H

#include <string>
#include <vector>

class DeviceDictionary final {
    char** hostPointersArray = nullptr;

    char** devicePointersArray = nullptr;
    size_t* devicePointersArraySize = nullptr;
public:
    DeviceDictionary(const std::vector<std::string>& words) {
        hostPointersArray = static_cast<char**>(malloc(words.size() * sizeof(char*)));
        if(hostPointersArray == nullptr)
            throw std::runtime_error("Host memory allocation failed for pointers array.");

        for(auto i = 0; i < words.size(); ++i) {
            size_t elemSize = 0;
        }
    }
};

#endif //HASHSELECTION_DEVICEDICTIONARY_H
