#include <iostream>
#include "DeviceHash.h"
#include "HostHash.h"

int main() {
    std::cout << HostSHA256("hello", 5).to_string() << std::endl;
    std::cout << DeviceSHA256("hello", 5).to_string() << std::endl;
}