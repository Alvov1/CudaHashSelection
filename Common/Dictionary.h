#ifndef HASHSELECTION_DICTIONARY_H
#define HASHSELECTION_DICTIONARY_H

#include <random>

#include "Word.h"
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

    const auto index = 10;//distribution(rng);
    return words[index];
}

template <typename Char>
const Word<Char>& Dictionary<Char>::get(unsigned int index) {
    if(index > size())
        throw std::invalid_argument("Index is out of range.");
    return words[index];
}

template <>
constexpr const Word<char> Dictionary<char>::words[] = {
        { "123456" }, { "123456789" }, { "12345" }, { "qwerty" }, { "password" }, 
        { "12345678" }, { "111111" }, { "123123" }, { "1234567890" }, { "1234567" }, 
        { "qwerty123" }, { "000000" }, { "1q2w3e" }, { "aa12345678" }, { "abc123" }, 
        { "password1" }, { "1234" }, { "qwertyuiop" }, { "123321" }, { "password123" },
        { "1q2w3e4r5t" }, { "iloveyou" }, { "654321" }, { "666666" }, { "987654321" },
        { "123" }, { "123456a" }, { "qwe123" }, { "1q2w3e4r" }, { "7777777" },
        { "1qaz2wsx" }, { "123qwe" }, { "zxcvbnm" }, { "121212" }, { "asdasd" },
        { "a123456" }, { "555555" }, { "dragon" }, { "112233" }, { "123123123" },
        { "monkey" }, { "11111111" }, { "qazwsx" }, { "159753" }, { "asdfghjkl" },
        { "222222" }, { "1234qwer" }, { "qwerty1" }, { "123654" }, { "123abc" },
        { "asdfgh" }, { "777777" }, { "aaaaaa" }, { "myspace1" }, { "88888888" },
        { "fuckyou" }, { "123456789a" }, { "999999" }, { "888888" }, { "football" },
        { "princess" }, { "789456123" }, { "147258369" }, { "1111111" }, { "sunshine" },
        { "michael" }, { "computer" }, { "qwer1234" }, { "daniel" }, { "789456" },
        { "11111" }, { "abcd1234" }, { "q1w2e3r4" }, { "shadow" }, { "159357" },
        { "123456q" }, { "1111" }, { "samsung" }, { "killer" }, { "asd123" },
        { "superman" }, { "master" }, { "12345a" }, { "azerty" }, { "zxcvbn" },
        { "qazwsxedc" }, { "131313" }, { "ashley" }, { "target123" }, { "987654" },
        { "baseball" }, { "qwert" }, { "asdasd123" }, { "qwerty" }, { "soccer" },
        { "charlie" }, { "qweasdzxc" }, { "tinkle" }, { "jessica" }, { "q1w2e3r4t5" },
        { "asdf" }, { "test1" }, { "1g2w3e4r" }, { "gwerty123" }, { "zag12wsx" },
        { "gwerty" }, { "147258" }, { "12341234" }, { "qweqwe" }, { "jordan" },
        { "pokemon" }, { "q1w2e3r4t5y6" }, { "12345678910" }, { "1111111111" }, { "12344321" },
        { "thomas" }, { "love" }, { "12qwaszx" }, { "102030" }, { "welcome" },
        { "liverpool" }, { "iloveyou1" }, { "michelle" }, { "101010" }, { "1234561" },
        { "hello" }, { "andrew" }, { "a123456789" }, { "a12345" }, { "Status" },
        { "fuckyou1" }, { "1qaz2wsx3edc" }, { "hunter" }, { "princess1" }, { "naruto" },
        { "justin" }, { "jennifer" }, { "qwerty12" }, { "qweasd" }, { "anthony" },
        { "andrea" }, { "joshua" }, { "asdf1234" }, { "12345qwert" }, { "1qazxsw2" },
        { "marina" }, { "love123" }, { "111222" }, { "robert" }, { "10203" },
        { "nicole" }, { "letmein" }, { "football1" }, { "secret" }, { "1234554321" },
        { "freedom" }, { "michael1" }, { "11223344" }, { "qqqqqq" }, { "123654789" },
        { "chocolate" }, { "12345q" }, { "internet" }, { "q1w2e3" }, { "google" },
        { "starwars" }, { "mynoob" }, { "qwertyui" }, { "55555" }, { "qwertyu" },
        { "lol123" }, { "lovely" }, { "monkey1" }, { "nikita" }, { "pakistan" },
        { "7758521" }, { "87654321" }, { "147852" }, { "jordan23" }, { "212121" },
        { "123789" }, { "147852369" }, { "123456789q" }, { "qwe" }, { "forever" },
        { "741852963" }, { "123qweasd" }, { "123456abc" }, { "1q2w3e4r5t6y" }, { "qazxsw" },
        { "456789" }, { "232323" }, { "999999999" }, { "qwerty12345" }, { "qwaszx" },
        { "1234567891" }, { "456123" }, { "444444" }, { "qq123456" }, { "xxx" }
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
    }
}

template<typename Char>
void Dictionary<Char>::calculateQuantities() {
    Console::out << "The calculations involve iterating over the following number of permutations:" << Console::endl;
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
    Console::out << "In total: " << total << " variations." << Console::endl << Console::endl;
}

#endif //HASHSELECTION_DICTIONARY_H
