#ifndef HASHSELECTION_DICTIONARY_H
#define HASHSELECTION_DICTIONARY_H

#include <random>

#include "Word.h"
#include "HostHash.h"
#include "ReplacementDictionary.h"

template <typename Char = char>
class Dictionary final {
    static const Word<Char> words[];
    Dictionary() = default;
public:
    static size_t size();
    static const Word<Char>& get(unsigned index);
    static const Word<Char>& getRandom();
    static void find(const std::basic_string<Char>& hash);
    static void calculateQuantities();

    Dictionary(const Dictionary& copy) = delete;
    Dictionary& operator=(const Dictionary& assign) = delete;
    Dictionary(Dictionary&& move) = delete;
    Dictionary& operator=(Dictionary&& moveAssign) = delete;
};

template <typename Char>
const Word<Char>& Dictionary<Char>::getRandom() {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> distribution(0, size() - 1);

    const auto index = distribution(rng);
    return Dictionary<Char>::words[index];
}

template <typename Char>
const Word<Char>& Dictionary<Char>::get(unsigned int index) {
    if(index > size())
        throw std::invalid_argument("Index is out of range.");
    return Dictionary<Char>::words[index];
}

template <>
constexpr const Word<char> Dictionary<char>::words[] = {
        { "123456" }, { "123456789" }, { "12345" }, { "qwerty" }, { "password" },
        { "12345678" }, { "111111" }, { "123123" }, { "1234567890" }, { "1234567" },
        { "qwerty123" }, { "000000" }, { "1q2w3e" }, { "aa12345678" }, { "abc123" },
        { "password1" }, { "1234" }, { "qwertyuiop" }, { "123321" }, { "password123" }
};

template <>
constexpr const Word<wchar_t> Dictionary<wchar_t>::words[] = {
        { L"123456" }, { L"123456789" }, { L"12345" }, { L"qwerty" }, { L"password" },
        { L"12345678" }, { L"111111" }, { L"123123" }, { L"1234567890" }, { L"1234567" },
        { L"qwerty123" }, { L"000000" }, { L"1q2w3e" }, { L"aa12345678" }, { L"abc123" },
        { L"password1" }, { L"1234" }, { L"qwertyuiop" }, { L"123321" }, { L"password123" }
};

template <typename Char>
size_t Dictionary<Char>::size() {
    return sizeof(words) / sizeof(Word<Char>);
}

template<typename Char>
void Dictionary<Char>::find(const std::basic_string<Char>& hash) {
    static typename ReplacementDictionary<Char>::Comparator closure =
            [](const std::basic_string<Char>& current, const std::basic_string<Char>& requiredHash) {
                HostSHA256 currentHash(current.c_str(), current.size());
                return currentHash.to_string() == requiredHash;
            };

    for(unsigned i = 0; i < Dictionary<Char>::size(); ++i) {
        const Word<Char>& current = Dictionary<Char>::get(i);

        const auto result = ReplacementDictionary<Char>::enumerate(current, hash, closure);
        if(result.has_value()) {
            Console::timer << "Found a coincidence with word " << result.value() << L"." << Console::endl;
            break;
        }
        Console::timer << "Word " << current << " completed." << Console::endl;
    }
}

template<typename Char>
void Dictionary<Char>::calculateQuantities() {
    static constexpr unsigned approximateFrequency = 15000;

    Console::out << "The calculations involve iterating over the following number of permutations for each word:" << Console::endl;
    unsigned long long total = 0;
    for(unsigned i = 0; i < Dictionary<Char>::size(); ++i) {
        unsigned long long tCount = 1;
        const Word<Char>& current = Dictionary<Char>::get(i);
        for(unsigned j = 0; j < current.size(); ++j)
            tCount *= (ReplacementDictionary<Char>::getVariants(current[j]).size() + 1);
        Console::out << i + 1 << ". " << current << " - " << tCount << Console::endl;
        total += tCount;
    }
    Console::out << "-----------------------------------------------------------------------------" << Console::endl;
    Console::out << "In total: " << total << " variations." << Console::endl;
    Console::out << "Exhaustive CPU search would take approximately ";
    const auto duration = std::chrono::duration_cast<std::chrono::minutes>(
            std::chrono::seconds(static_cast<unsigned>(static_cast<double>(total) / static_cast<double>(approximateFrequency))));
    Console::out << duration.count() << " minutes." << Console::endl << Console::endl;

}

#endif //HASHSELECTION_DICTIONARY_H
