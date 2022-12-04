#ifndef HASHSELECTION_PASSWORDDICTIONARY_H
#define HASHSELECTION_PASSWORDDICTIONARY_H

#include <vector>
#include <random>

#include "Timer.h"
#include "Console.h"

#include "Dictionary.h"
#include "MutationDictionary.h"

class PasswordDictionary final : public IDictionary {
public:
    const WordArray& get() const override;
    const std::string& getRandom() const;
    const std::string& operator[](unsigned index) const { return get()[index]; };

    void find(const MutationDictionary& replacements, const std::string& requiredValue, const Comparator& closure) const;
    void calculateQuantities(const MutationDictionary& replacements) const;

    static unsigned upperPower2(unsigned value);
};

#endif //HASHSELECTION_PASSWORDDICTIONARY_H
