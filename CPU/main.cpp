#include "ScreenWriter.h"
#include "Dictionary.h"
#include "HostHash.h"

int main() {
    using Char = char;
    ReplacementDictionary<Char>::showVariants();
    Dictionary<Char>::calculateQuantities();

    const auto plainPassword = Dictionary<Char>::getRandom();
    Console::timer << "Using word: " << plainPassword << Console::endl;

    const auto mutatedPassword = ReplacementDictionary<Char>::rearrange(plainPassword);
    Console::timer << "Using after rearrangement: " << mutatedPassword << Console::endl;

    const HostSHA256 hash(mutatedPassword.c_str(), mutatedPassword.size());
    Console::timer << "Searching for mutatedPassword with hash '" << hash.to_string() << "'." << Console::endl;

    Dictionary<Char>::find(hash.to_string());
    return 0;
}