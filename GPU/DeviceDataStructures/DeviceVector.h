#ifndef HASHSELECTION_DEVICEVECTOR_H
#define HASHSELECTION_DEVICEVECTOR_H

#include <cstdint>
#define DEVICE __device__

template <typename T>
class DeviceVector final {
    T* buffer = nullptr;
    size_t filled = 0;
    size_t capacity = 0;

    DEVICE static int strcmp(const char *s1, const char *s2) {
        for (; *s1 == *s2; s1++, s2++)
            if (*s1 == '\0')
                return 0;
        return ((*(unsigned char *) s1 < *(unsigned char *) s2) ? -1 : +1);
    }
public:
    DEVICE DeviceVector(size_t size)
    : buffer(static_cast<T*>(malloc((size + 1) * sizeof(T)))),
      filled(0), capacity(size + 1) { buffer[capacity - 1] = T(); };

    DEVICE DeviceVector(size_t size, const T& value)
    : buffer(static_cast<T*>(malloc((size + 1) * sizeof(T)))),
      filled(size), capacity(size + 1) {
        for(unsigned i = 0; i < size; ++i)
            buffer[i] = value;
        buffer[capacity + 1] = T();
    };

    DEVICE DeviceVector(const T* ptr, size_t size)
    : buffer(static_cast<T*>(malloc((size + 1) * sizeof(T)))),
      filled(size), capacity(size + 1) {
        memcpy(buffer, ptr, size * sizeof(T));
        buffer[capacity] = T();
    }

    DEVICE void push_back(const T& value) {
        if(filled < capacity)
            buffer[filled++] = value;
        else
            printf("Subscript index is out of range (push_back()).\n");
    }
    DEVICE T pop_back() {
        if(filled > 0)
            return buffer[filled--];
        printf("Subscript index is out of range (pop_back()).\n");
    }

    DEVICE size_t size() const { return filled; }
    DEVICE T& operator[](size_t index) const {
        if(index < filled)
            return buffer[index];
        printf("Subscript index is out of range (operator []).\n");
        return buffer[0];
    }
    DEVICE inline const T* get() const { return buffer; }
    DEVICE T* getInternalCopy() const {
        return static_cast<T*>(malloc(sizeof(T) * capacity));
    }

    DEVICE ~DeviceVector() { free(buffer); }
    DEVICE DeviceVector(const DeviceVector& copy) = delete;
    DEVICE DeviceVector& operator=(const DeviceVector& copy) = delete;
    DEVICE DeviceVector(DeviceVector&& move) noexcept = default;
    DEVICE DeviceVector& operator=(DeviceVector&& move) noexcept = default;

    friend class DeviceWord;
};

#endif //HASHSELECTION_DEVICEVECTOR_H
