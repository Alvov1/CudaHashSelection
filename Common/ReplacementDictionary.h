#ifndef PROJECT2_REPLACEMENTDICTIONARY_H
#define PROJECT2_REPLACEMENTDICTIONARY_H

#include <iostream>
#include <string>
#include <random>
#include <optional>

#include "Timer.h"
#include "Console.h"
#include "Dictionary.h"

class ReplacementDictionary final: public IDictionary {
public:
    const WordArray& get() const override;
    const std::string& operator[](char key) const;
    void show() const;

    std::string rearrange(const std::string& word) const;

    std::optional<std::string> enumerate(const std::string& candidate,
         const std::string& pattern, const Comparator& func) const;
private:
    bool nextPermutation(const std::string& candidate,
         const std::string& pattern, std::string& buffer, const Comparator& func) const;
};

#endif //PROJECT2_REPLACEMENTDICTIONARY_H
