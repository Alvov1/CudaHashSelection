#include <iostream>

#include "Timer.h"
#include "HostHash.h"
#include "Dictionary.h"

void process(const std::string& hash) {
    for(auto i = 0; i < Dictionary::size(); ++i) {
        HostSHA256 sha(Dictionary::data()[i]);
        if(sha.out() == hash) {
            Timer::out << "Found coincidence at " << i << ". Password is " << Dictionary::instance()[i] << std::endl;
            break;
        }
    }
}

int main() {
    const auto pass = Dictionary::giveRandom();
    const auto hash = HostSHA256(pass).out();
    Timer::out << "Searching for password with hash '" << hash << "'." << std::endl;

    process(hash);
    return 0;
}