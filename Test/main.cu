#include <iostream>
#include <thrust/pair.h>
#include <thrust/device_vector.h>

__global__ void run() {

}

int main() {
    const std::vector<thrust::pair<unsigned, unsigned>> data = { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 5 } };
    const thrust::device_vector<thrust::pair<unsigned, unsigned>> deviceData = data;
}