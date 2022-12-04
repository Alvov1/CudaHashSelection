#ifndef HASHSELECTION_DEVICESTACK_H
#define HASHSELECTION_DEVICESTACK_H

template <typename Value, unsigned size>
class DeviceStack final {
    Value buffer[size] {};
    size_t filled = 0;
public:
    DeviceStack() = default;

    void push(const Value& value) {
        if(filled < size)
            buffer[filled++] = value;
        else
            printf("Stack subscript index is out of range.");
    }
    void pop() {
        if(filled > 0)
            filled--;
        else
            printf("Stack is empty.");
    }
    Value top() {
        if(filled > 0)
            return buffer[filled];
        else
            printf("Stack is empty.");
        return Value();
    }

    const Value* get() const { return &buffer; }
    size_t size() const { return size; }

    ~DeviceStack() = default;
    DeviceStack(const DeviceStack& copy) = default;
    DeviceStack& operator=(const DeviceStack& assign) = default;
    DeviceStack(DeviceStack&& move) = default;
    DeviceStack& operator=(DeviceStack&& moveAssign) = default;
};

#endif //HASHSELECTION_DEVICESTACK_H
