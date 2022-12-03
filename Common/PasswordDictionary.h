#ifndef HASHSELECTION_PASSWORDDICTIONARY_H
#define HASHSELECTION_PASSWORDDICTIONARY_H

#include <vector>
#include <random>

#include "Timer.h"
#include "Console.h"
#include "Dictionary.h"
#include "ReplacementDictionary.h"

class PasswordDictionary final : public IDictionary {
public:
    const WordArray& get() const override;
    const std::string& operator[](unsigned index) const { return get()[index]; };
    const std::string& getRandom() const;;

    void find(const ReplacementDictionary& replacements, const std::string& requiredValue, const Comparator& closure) const;
    void calculateQuantities(const ReplacementDictionary& replacements) const;
};

#endif //HASHSELECTION_PASSWORDDICTIONARY_H
