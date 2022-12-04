#ifndef HASHSELECTION_DEVICESTACK_H
#define HASHSELECTION_DEVICESTACK_H

template <typename Value>
class DeviceStack final {
    Value* buffer = nullptr;
    size_t filled = 0;
    const size_t length = 0;
public:
    DEVICE DeviceStack() = default;
    DEVICE DeviceStack(size_t size)
    : buffer(static_cast<Value*>(malloc(sizeof(Value) * size))), length(size) {};

    DEVICE void push(const Value& value) {
        if(filled < length)
            buffer[filled++] = value;
        else
            printf("Stack subscript index is out of range.");
    }
    DEVICE void pop() {
        if(filled > 0)
            filled--;
        else
            printf("Stack is empty.");
    }
    DEVICE Value top() {
        if(filled > 0)
            return buffer[filled - 1];
        printf("Stack is empty.");
        return Value();
    }

    DEVICE const Value* const get() const { return &buffer; }
    DEVICE size_t size() const { return filled; }
    DEVICE bool empty() const { return filled == 0; }
    DEVICE const Value& operator[](unsigned index) const {
        if(index < filled)
            return buffer[index];
        printf("Stack subscript is out of range.");
        return buffer[0];
    }

    DEVICE ~DeviceStack() { free(buffer); }
    DEVICE DeviceStack(const DeviceStack& copy) = default;
    DEVICE DeviceStack& operator=(const DeviceStack& assign) = default;
    DEVICE DeviceStack(DeviceStack&& move) noexcept = default;
    DEVICE DeviceStack& operator=(DeviceStack&& moveAssign) noexcept = default;
};

#endif //HASHSELECTION_DEVICESTACK_H
