#include "HashSelection.h"

std::wostream& operator<<(std::wostream& stream, const HashSelection::Word& word) {
    return stream << word.first.data();
}

int main() {
    const HashSelection::Word word = { { 'r', 'o', 'o', 'o', 'l', 'e', 'r' }, 7 };
    for(const auto& word2: HashSelection::foundExtensions(word))
        std::wcout << word2 << std::endl;

    return 0;
}