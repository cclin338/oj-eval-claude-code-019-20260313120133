#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // For iteration i (0-indexed, but represents round i+1):
    // Q.shape = [i+1, 512], K[j].shape = [1, 512], V[j].shape = [1, 512]
    // We need to compute attention for Q with K[0...i] and V[0...i]

    // Move current_query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Step 1: Build concatenated K matrix in HBM first (concat is faster in HBM than SRAM)
    Matrix *keys_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        keys_all = matrix_memory_allocator.Allocate("keys_all");
        gpu_sim.Copy(keys[j], keys_all, kInGpuHbm);
      } else {
        Matrix *new_keys = matrix_memory_allocator.Allocate("keys_all");
        gpu_sim.Concat(keys_all, keys[j], new_keys, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(keys_all);
        keys_all = new_keys;
      }
    }

    // Move to SRAM and transpose: [i+1, 512] -> [512, i+1]
    gpu_sim.MoveMatrixToSharedMem(keys_all);
    gpu_sim.Transpose(keys_all, kInSharedMemory);

    // Step 2: Compute Q @ K^T in one operation: [i+1, 512] @ [512, i+1] = [i+1, i+1]
    Matrix *all_scores = matrix_memory_allocator.Allocate("all_scores");
    gpu_sim.MatMul(current_query, keys_all, all_scores);
    gpu_sim.ReleaseMatrix(keys_all);

    // Step 3: Apply softmax row-wise
    Matrix *all_exp = matrix_memory_allocator.Allocate("all_exp");
    gpu_sim.MatExp(all_scores, all_exp);
    gpu_sim.ReleaseMatrix(all_scores);

    // Compute row sums and normalize
    Matrix *softmax_matrix = nullptr;

    for (size_t row = 0; row <= i; ++row) {
      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.GetRow(all_exp, row, exp_row, kInSharedMemory);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(exp_row, row_sum);

      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(exp_row, row_sum, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (row == 0) {
        softmax_matrix = softmax_row;
      } else {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_matrix");
        gpu_sim.Concat(softmax_matrix, softmax_row, new_softmax, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_matrix);
        gpu_sim.ReleaseMatrix(softmax_row);
        softmax_matrix = new_softmax;
      }
    }

    gpu_sim.ReleaseMatrix(all_exp);

    // Step 4: Build concatenated V matrix
    Matrix *values_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (values[j]->GetPosition() != kInSharedMemory) {
        gpu_sim.MoveMatrixToSharedMem(values[j]);
      }

      if (j == 0) {
        values_all = matrix_memory_allocator.Allocate("values_all");
        gpu_sim.Copy(values[j], values_all, kInSharedMemory);
      } else {
        Matrix *new_values = matrix_memory_allocator.Allocate("values_all");
        gpu_sim.Concat(values_all, values[j], new_values, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(values_all);
        values_all = new_values;
      }
    }

    // Step 5: Compute softmax_matrix @ values_all: [i+1, i+1] @ [i+1, 512] = [i+1, 512]
    Matrix *result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_matrix, values_all, result);
    gpu_sim.ReleaseMatrix(softmax_matrix);
    gpu_sim.ReleaseMatrix(values_all);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu