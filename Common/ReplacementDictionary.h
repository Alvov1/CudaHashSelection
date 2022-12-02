#ifndef PROJECT2_REPLACEMENTDICTIONARY_H
#define PROJECT2_REPLACEMENTDICTIONARY_H

#include <iostream>
#include <string>
#include <random>
#include <optional>

#include "Console.h"
#include "Dictionary.h"

class ReplacementDictionary final: public IDictionary {
    const WordArray& get() const override;
public:
    const std::string& operator[](char key) const { return get()[std::tolower(key) - 'a']; };
    void show() const;

    std::string rearrange(const std::string& word) const;

    std::optional<std::string> enumerate(const std::string& candidate,
         const std::string& pattern, const Comparator& func) const;
private:
    bool nextPermutation(const std::string& candidate,
         const std::string& pattern, std::string& buffer, const Comparator& func) const;
};

#endif //PROJECT2_REPLACEMENTDICTIONARY_H
