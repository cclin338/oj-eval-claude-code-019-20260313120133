#include "simulator.hpp"
#include <iostream>

int main() {
    sjtu::GpuSimulator gpu_sim;
    sjtu::MatrixMemoryAllocator alloc;

    // Create matrices: Q=[2,512], K=[1,512]
    std::vector<float> query_data(2*512, 1.0f);
    std::vector<float> key_data(512, 2.0f);

    sjtu::Matrix *query = new sjtu::Matrix(2, 512, query_data, gpu_sim);
    sjtu::Matrix *key = new sjtu::Matrix(1, 512, key_data, gpu_sim);
    alloc.Bind(query, "query");
    alloc.Bind(key, "key");

    std::cerr << "Initial: query=" << query->GetRowNum() << "x" << query->GetColumnNum()
              << " key=" << key->GetRowNum() << "x" << key->GetColumnNum() << std::endl;

    // Move both to SRAM
    gpu_sim.MoveMatrixToSharedMem(query);
    gpu_sim.MoveMatrixToSharedMem(key);

    // Copy key
    sjtu::Matrix *key_copy = alloc.Allocate("key_copy");
    gpu_sim.Copy(key, key_copy, sjtu::kInSharedMemory);

    // Transpose key_copy
    gpu_sim.Transpose(key_copy, sjtu::kInSharedMemory);

    // Multiply query @ key_copy^T
    sjtu::Matrix *result = alloc.Allocate("result");
    gpu_sim.MatMul(query, key_copy, result);

    // Run simulator
    std::cerr << "Running simulator..." << std::endl;
    gpu_sim.Run(true, &alloc);

    std::cerr << "After run: query=" << query->GetRowNum() << "x" << query->GetColumnNum()
              << " key_copy=" << key_copy->GetRowNum() << "x" << key_copy->GetColumnNum()
              << " result=" << result->GetRowNum() << "x" << result->GetColumnNum() << std::endl;

    return 0;
}
