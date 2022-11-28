#include <functional>

#include "ScreenWriter.h"
#include "HostHash.h"
#include "Dictionary.h"

int main() {
    using Char = char;
    Dictionary<Char>::calculateQuantities();

    ReplacementDictionary<Char>::showVariants();

    const auto plainPassword = Dictionary<Char>::getRandom();
    Console::timer << "Using word: " << plainPassword << Console::endl;

    const auto mutatedPassword = ReplacementDictionary<Char>::rearrange(plainPassword);
    Console::timer << "Using after rearrangement: " << mutatedPassword << Console::endl;

    const HostSHA256 hash(mutatedPassword.c_str(), mutatedPassword.size());
    Console::timer << "Searching for mutatedPassword with hash '" << hash.to_string() << "'." << Console::endl;

    Dictionary<Char>::find(hash.to_string());
    return 0;
}