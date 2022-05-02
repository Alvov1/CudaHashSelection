#ifndef HASHSELECTION_ARRAY_H
#define HASHSELECTION_ARRAY_H

#include <cstdint>

DEVICE inline int devStrcmp(const char *s1, const char *s2) {
    for (; *s1 == *s2; s1++, s2++)
        if (*s1 == '\0')
            return 0;
    return ((*(unsigned char *) s1 < *(unsigned char *) s2) ? -1 : +1);
}

template <typename T>
class Array final {
    T* data_ = nullptr;
    size_t size_ = 0;
public:
    DEVICE Array(size_t size): data_(static_cast<T*>(malloc((size + 1) * sizeof(T)))), size_(size) {};
    DEVICE Array(const T* ptr, size_t size) : data_(static_cast<T*>(malloc((size + 1) * sizeof(T)))), size_(size) {
        for(auto i = 0; i < size_; i++)
            data_[i] = ptr[i];
        data_[size] = T();
    }
    DEVICE ~Array() { free(data_); size_ = 0; }

    DEVICE inline Array(const Array& copy) : Array(copy.data_, copy.size_) {};
    DEVICE Array& operator=(const Array& copy) {
        if(&copy == this) return *this;
        this->~Array();
        size_ = copy.size_;
        data_ = static_cast<T*>(malloc((size_ + 1) * sizeof(T)));
        for(auto i = 0; i < size_; ++i)
            data_[i] = copy.data_[i];
        data_[size_] = T();
        return *this;
    }

    DEVICE inline Array(Array&& move) noexcept {
        this->operator=(move);
    }
    DEVICE Array& operator=(Array&& move) noexcept {
        T* tPtr = data_; data_ = move.data_; move.data_ = tPtr;
        auto tSize = size_; size_ = move.size_; move.size_ = tSize;
        return *this;
    }

    DEVICE T& operator[](size_t index) const {
        if(index < size_)
            return data_[index];
        return data_[0];
    }
    DEVICE inline const T* get() const { return data_; }
    DEVICE inline bool compare(const T* cmp) const {
        return (devStrcmp(data_, cmp) == 0);
    }
};

#endif //HASHSELECTION_ARRAY_H
