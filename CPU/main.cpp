#include <iostream>
#include <functional>

#include "Timer.h"
#include "HostHash.h"
#include "Dictionary.h"
#include "ReplacementDictionary.h"

void process(const Word& hash) {
    static std::function<bool(const Word&, const Word&)> closure = [](const Word& current, const Word& requiredHash) {
        HostSHA256 currentHash(current);
        return currentHash == requiredHash;
    };

    for(unsigned i = 0; i < Dictionary::size(); ++i) {
        const Word current = Dictionary::get(i);

        ReplacementDictionary::enumerate(current, hash, closure);
    }
}

int main() {
    const Word pass = Dictionary::getRandom();
    const HostSHA256 hash(pass);
    Timer::out << L"Searching for password with hash '" << hash << L"'." << std::endl;

    process(hash.toWord());
    return 0;
}