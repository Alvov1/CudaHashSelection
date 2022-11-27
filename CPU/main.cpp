#include <iostream>
#include <functional>
#include <Windows.h>

#include "Timer.h"
//#include "HostHash.h"
//#include "Dictionary.h"
#include "ReplacementDictionary.h"

//void process(const std::wstring& hash) {
//    static std::function<bool(const Word&, const std::wstring&)> closure = [](const Word& current, const std::wstring& requiredHash) {
//        HostSHA256 currentHash(current.c_str(), current.size());
//        return currentHash.to_wstring() == requiredHash;
//    };
//
//    for(unsigned i = 0; i < Dictionary::size(); ++i) {
//        const Word current = Dictionary::get(i);
//
//        const std::optional<Word> result = ReplacementDictionary::enumerate(current, hash, closure);
//        if(result.has_value())
//            Timer::out << "Found a coincidence with word " << result.value() << L"." << Timer::endl;
//    }
//}

int main() {
    using Char = char;

    ReplacementDictionary<Char>::getVariants('A');

//    const auto password = ReplacementDictionary<Char>::rearrange(Dictionary::getRandom());
//    Timer::out << L"Using word " << password << std::Endliner;
//
//    const HostSHA256 hash(password.c_str(), password.size());
//    Timer::out << L"Searching for password with hash '" << hash.to_wstring() << L"'." << std::Endliner;
//
//    process(hash.to_wstring());
    return 0;
}