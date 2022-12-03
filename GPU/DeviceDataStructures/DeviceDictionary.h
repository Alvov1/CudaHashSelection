#ifndef HASHSELECTION_DEVICEDICTIONARY_H
#define HASHSELECTION_DEVICEDICTIONARY_H

#include <string>
#include <vector>
#include <memory>

class DeviceDictionary final {
    std::unique_ptr<char*> hostPointersArray = nullptr;
    size_t hostPointersArraySize = 0;
    char** devicePointersArray = nullptr;
public:
    DeviceDictionary(const std::vector<std::string>& words);
    std::pair<char**, size_t> get();
    ~DeviceDictionary();
};

#endif //HASHSELECTION_DEVICEDICTIONARY_H
