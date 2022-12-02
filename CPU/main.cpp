#include "Timer.h"
#include "HostHash.h"
#include "PasswordDictionary.h"

int main() {
    ReplacementDictionary replacements;
    replacements.show();

    PasswordDictionary passwords;
    passwords.calculateQuantities(replacements);

    const auto& plainPassword = passwords.getRandom();
    Console::timer << "Using word: " << plainPassword << Console::endl;

    const auto mutatedPassword = replacements.rearrange(plainPassword);
    Console::timer << "Using after rearrangement: " << mutatedPassword << Console::endl;

    const HostSHA256 hash(mutatedPassword.c_str(), mutatedPassword.size());
    Console::timer << "Searching for mutatedPassword with hash '" << hash.to_string() << "'." << Console::endl;

    std::function comparator =
        [](const std::string& current, const std::string& requiredHash) {
            HostSHA256 currentHash(current.c_str(), current.size());
            return currentHash.to_string() == requiredHash;
        };

    passwords.find(replacements, hash.to_string(), comparator);
    return 0;
}