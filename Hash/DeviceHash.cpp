#include "DeviceHash.h"

//template<typename Char>
//DeviceSHA256<Char>::DeviceSHA256(const Char* data, std::size_t length) {
//
//}
//template DeviceSHA256<char>::DeviceSHA256(const char* data, std::size_t length);
//template DeviceSHA256<wchar_t>::DeviceSHA256(const wchar_t* data, std::size_t length);

DeviceSHA256::u32 DeviceSHA256::Substitutions::rotateRight(DeviceSHA256::u32 value, DeviceSHA256::u32 bits) {
    return (value >> bits) | (value << (32 - bits));
}

DeviceSHA256::u32 DeviceSHA256::Substitutions::choose(DeviceSHA256::u32 first, DeviceSHA256::u32 second, DeviceSHA256::u32 third) {
    return (first & second) ^ (~first & third);
}

DeviceSHA256::u32 DeviceSHA256::Substitutions::majority(DeviceSHA256::u32 first, DeviceSHA256::u32 second, DeviceSHA256::u32 third) {
    return (first & (second | third)) | (second & third);
}

DeviceSHA256::u32 DeviceSHA256::Substitutions::sig0(DeviceSHA256::u32 value) {
    return rotateRight(value, 7) ^ rotateRight(value, 18) ^ (value >> 3);
}

DeviceSHA256::u32 DeviceSHA256::Substitutions::sig1(DeviceSHA256::u32 value) {
    return rotateRight(value, 17) ^ rotateRight(value, 19) ^ (value >> 10);
}