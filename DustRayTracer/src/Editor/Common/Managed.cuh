#pragma once
#include <cuda_runtime.h>

class Managed {
public:
    void* operator new(size_t len) {
        void* ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

