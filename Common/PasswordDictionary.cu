#include "PasswordDictionary.h"

const IDictionary::WordArray &PasswordDictionary::get() const {
    static std::vector<std::string> words {
            {"123456"}, {"123456789"}, {"12345"}, {"qwerty"}, {"password"},
            {"12345678"}, {"111111"}, {"123123"}, {"1234567890"}, {"1234567"},
            {"qwerty123"}, {"000000"}, {"1q2w3e"}, {"aa12345678"}, {"abc123"},
            {"password1"}, {"1234"}, {"qwertyuiop"}, {"123321"}, {"password123"} };

    return words;
}

void PasswordDictionary::find(const ReplacementDictionary& replacements, const std::basic_string<char>& hash, const Comparator& closure) const {
    for(unsigned i = 0; i < replacements.size(); ++i) {
        const auto& current = this->operator[](i);

        const auto result = replacements.enumerate(current, hash, closure);
        if(result.has_value()) {
            Console::timer << "Found a coincidence with word " << result.value() << L"." << Console::endl;
            break;
        }
        Console::timer << "Word " << current << " completed." << Console::endl;
    }
}

void PasswordDictionary::calculateQuantities(const ReplacementDictionary& replacements) const {
    static constexpr unsigned approximateFrequency = 15000;

    Console::cout << "The calculations involve iterating over the following number of permutations for each word:" << Console::endl;
    unsigned long long total = 0;
    for(const auto& current: this->get()) {
        static unsigned short number = 1;
        unsigned long long tCount = 1;
        for(char j : current)
            tCount *= (replacements[j].size() + 1);
        Console::cout << number++ << ". " << current << " - " << tCount << Console::endl;
        total += tCount;
    }
    Console::cout << "-----------------------------------------------------------------------------" << Console::endl;
    Console::cout << "In total: " << total << " variations." << Console::endl;
    Console::cout << "Exhaustive CPU search would take approximately ";
    const auto duration = std::chrono::duration_cast<std::chrono::minutes>(
            std::chrono::seconds(static_cast<unsigned>(static_cast<double>(total) / static_cast<double>(approximateFrequency))));
    Console::cout << duration.count() << " minutes." << Console::endl << Console::endl;
}

const std::string& PasswordDictionary::getRandom() const {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> distribution(0, size() - 1);

    const auto index = distribution(rng);
    return this->operator[](index);
}

