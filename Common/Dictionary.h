#ifndef HASHSELECTION_DICTIONARY_H
#define HASHSELECTION_DICTIONARY_H

#include <vector>
#include <string>

class IDictionary {
protected:
    using WordArray = std::vector<std::string>;
    virtual const WordArray& get() const = 0;
    static const std::string emptyWord;

    using Comparator = std::function<bool(const std::string&, const std::string&)>;
public:
    size_t size() const { return get().size(); }

    IDictionary() = default;
    virtual ~IDictionary() = default;
    IDictionary(const IDictionary& copy) = delete;
    IDictionary& operator=(const IDictionary& assign) = delete;
    IDictionary(IDictionary&& move) = delete;
    IDictionary& operator=(IDictionary&& moveAssign) = delete;
};

#endif //HASHSELECTION_DICTIONARY_H
