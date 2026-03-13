#include "simulator.hpp"
#include <iostream>

int main() {
    sjtu::GpuSimulator gpu_sim;
    sjtu::MatrixMemoryAllocator alloc;

    // Create Q=[2,512], K[0]=[1,512], K[1]=[1,512], V[0]=[1,512], V[1]=[1,512]
    std::vector<float> query_data(2*512, 0.01f);
    std::vector<float> key0_data(512, 1.0f);
    std::vector<float> key1_data(512, 2.0f);
    std::vector<float> value0_data(512, 0.5f);
    std::vector<float> value1_data(512, 1.5f);

    sjtu::Matrix *query = new sjtu::Matrix(2, 512, query_data, gpu_sim);
    sjtu::Matrix *keys[2];
    sjtu::Matrix *values[2];
    keys[0] = new sjtu::Matrix(1, 512, key0_data, gpu_sim);
    keys[1] = new sjtu::Matrix(1, 512, key1_data, gpu_sim);
    values[0] = new sjtu::Matrix(1, 512, value0_data, gpu_sim);
    values[1] = new sjtu::Matrix(1, 512, value1_data, gpu_sim);

    alloc.Bind(query, "query");
    alloc.Bind(keys[0], "key0");
    alloc.Bind(keys[1], "key1");
    alloc.Bind(values[0], "value0");
    alloc.Bind(values[1], "value1");

    // Move query to SRAM
    gpu_sim.MoveMatrixToSharedMem(query);

    sjtu::Matrix *result_sum = nullptr;

    // Process each key-value pair
    for (size_t j = 0; j < 2; ++j) {
        std::cerr << "Processing j=" << j << std::endl;

        // Move key and value to SRAM
        gpu_sim.MoveMatrixToSharedMem(keys[j]);
        gpu_sim.MoveMatrixToSharedMem(values[j]);

        // Copy and transpose key
        sjtu::Matrix *key_t = alloc.Allocate("key_t");
        gpu_sim.Copy(keys[j], key_t, sjtu::kInSharedMemory);
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

        // softmax
        sjtu::Matrix *softmax = alloc.Allocate("softmax");
        gpu_sim.MatDiv(exp_scores, sum_exp, softmax);
        gpu_sim.ReleaseMatrix(exp_scores);
        gpu_sim.ReleaseMatrix(sum_exp);

        // softmax @ value
        sjtu::Matrix *attention_output = alloc.Allocate("att_out");
        gpu_sim.MatMul(softmax, values[j], attention_output);
        gpu_sim.ReleaseMatrix(softmax);

        // Accumulate
        if (j == 0) {
            result_sum = attention_output;
        } else {
            sjtu::Matrix *new_sum = alloc.Allocate("new_sum");
            gpu_sim.MatAdd(result_sum, attention_output, new_sum);
            gpu_sim.ReleaseMatrix(result_sum);
            gpu_sim.ReleaseMatrix(attention_output);
            result_sum = new_sum;
        }

        // Move key and value back to HBM
        gpu_sim.MoveMatrixToGpuHbm(keys[j]);
        gpu_sim.MoveMatrixToGpuHbm(values[j]);
    }

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result_sum);

    // Run simulator
    std::cerr << "Running simulator..." << std::endl;
    gpu_sim.Run(false, &alloc);

    std::cerr << "Success! Result shape: " << result_sum->GetRowNum() << "x"
              << result_sum->GetColumnNum() << " Position=" << result_sum->GetPosition() << std::endl;

    return 0;
}
