#include "HashSelection.h"

std::wostream& operator<<(std::wostream& stream, const HashSelection::Word& word) {
    return stream << word.first.data();
}

int main() {
    HashSelection::Word word = {{ L'p', L'a', L's', L's', L'w', L'o', L'r', L'd' }, 8 };
    std::function<bool(const HashSelection::Word&)> closure = [] (const HashSelection::Word& word) {
        std::wcout << word << std::endl;
        return false;
    };
    HashSelection::foundPermutations(word, closure);

    return 0;
}