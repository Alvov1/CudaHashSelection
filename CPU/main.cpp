#include <functional>

#include "Timer.h"
#include "HostHash.h"
#include "Dictionary.h"

int main() {
    using Char = char;

    const auto password = ReplacementDictionary<Char>::rearrange(Dictionary<Char>::getRandom());
    Timer::out << "Using word " << password << Timer::endl;

    const HostSHA256 hash(password.c_str(), password.size());
    Timer::out << "Searching for password with hash '" << hash.to_string() << "'." << Timer::endl;

    Dictionary<Char>::find(hash.to_string());
    return 0;
}