#include "simulator.hpp"
#include <iostream>
#include <cmath>

int main() {
    sjtu::GpuSimulator gpu_sim;
    sjtu::MatrixMemoryAllocator alloc;

    // Create matrices: Q=[2,512], K=[1,512], V=[1,512]
    std::vector<float> query_data(2*512);
    for (size_t i = 0; i < query_data.size(); ++i) {
        query_data[i] = 0.01f * (i % 512);
    }
    std::vector<float> key_data(512);
    for (size_t i = 0; i < key_data.size(); ++i) {
        key_data[i] = 0.01f * i;
    }
    std::vector<float> value_data(512);
    for (size_t i = 0; i < value_data.size(); ++i) {
        value_data[i] = 1.0f;
    }

    sjtu::Matrix *query = new sjtu::Matrix(2, 512, query_data, gpu_sim);
    sjtu::Matrix *key = new sjtu::Matrix(1, 512, key_data, gpu_sim);
    sjtu::Matrix *value = new sjtu::Matrix(1, 512, value_data, gpu_sim);
    alloc.Bind(query, "query");
    alloc.Bind(key, "key");
    alloc.Bind(value, "value");

    // Move to SRAM
    gpu_sim.MoveMatrixToSharedMem(query);
    gpu_sim.MoveMatrixToSharedMem(key);
    gpu_sim.MoveMatrixToSharedMem(value);

    // Transpose key
    sjtu::Matrix *key_t = alloc.Allocate("key_t");
    gpu_sim.Copy(key, key_t, sjtu::kInSharedMemory);
    gpu_sim.Transpose(key_t, sjtu::kInSharedMemory);

    // Q @ K^T
    sjtu::Matrix *scores = alloc.Allocate("scores");
    gpu_sim.MatMul(query, key_t, scores);
    gpu_sim.ReleaseMatrix(key_t);

    // exp(scores)
    sjtu::Matrix *exp_scores = alloc.Allocate("exp_scores");
    gpu_sim.MatExp(scores, exp_scores);
    gpu_sim.ReleaseMatrix(scores);

    // sum(exp_scores)
    sjtu::Matrix *sum_exp = alloc.Allocate("sum_exp");
    gpu_sim.Sum(exp_scores, sum_exp);

    // softmax = exp_scores / sum_exp
    sjtu::Matrix *softmax = alloc.Allocate("softmax");
    gpu_sim.MatDiv(exp_scores, sum_exp, softmax);
    gpu_sim.ReleaseMatrix(exp_scores);
    gpu_sim.ReleaseMatrix(sum_exp);

    // softmax @ value
    sjtu::Matrix *output = alloc.Allocate("output");
    gpu_sim.MatMul(softmax, value, output);
    gpu_sim.ReleaseMatrix(softmax);

    // Run simulator
    std::cerr << "Running simulator..." << std::endl;
    gpu_sim.Run(false, &alloc);

    std::cerr << "Success! Output shape: " << output->GetRowNum() << "x"
              << output->GetColumnNum() << " Position=" << output->GetPosition() << std::endl;

    return 0;
}
