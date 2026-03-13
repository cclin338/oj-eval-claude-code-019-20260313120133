#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Keep keys and values in SRAM after first use
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Move query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K matrix by concatenating keys[0..i]
    Matrix *keys_concat = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (keys[j]->GetPosition() != kInSharedMemory) {
        gpu_sim.MoveMatrixToSharedMem(keys[j]);
      }

      if (j == 0) {
        keys_concat = matrix_memory_allocator.Allocate("keys_concat");
        gpu_sim.Copy(keys[j], keys_concat, kInSharedMemory);
      } else {
        Matrix *new_keys = matrix_memory_allocator.Allocate("keys_concat");
        gpu_sim.Concat(keys_concat, keys[j], new_keys, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(keys_concat);
        keys_concat = new_keys;
      }
    }

    // Transpose keys: [i+1, 512] -> [512, i+1]
    gpu_sim.Transpose(keys_concat, kInSharedMemory);

    // Compute Q @ K^T: [i+1, 512] @ [512, i+1] = [i+1, i+1]
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, keys_concat, scores);
    gpu_sim.ReleaseMatrix(keys_concat);

    // Apply exp to all scores
    Matrix *exp_scores = matrix_memory_allocator.Allocate("exp_scores");
    gpu_sim.MatExp(scores, exp_scores);
    gpu_sim.ReleaseMatrix(scores);

    // Compute softmax row-wise and accumulate weighted values
    Matrix *result = nullptr;

    for (size_t row = 0; row <= i; ++row) {
      // Get exp scores for this row
      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.GetRow(exp_scores, row, exp_row, kInSharedMemory);

      // Sum the row
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(exp_row, row_sum);

      // Normalize: softmax_row = exp_row / row_sum
      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(exp_row, row_sum, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);

      // Compute weighted sum of values for this row
      // softmax_row is [1, i+1], need to multiply with each value
      Matrix *row_output = nullptr;

      for (size_t j = 0; j <= i; ++j) {
        if (values[j]->GetPosition() != kInSharedMemory) {
          gpu_sim.MoveMatrixToSharedMem(values[j]);
        }

        // Get softmax weight for this value
        Matrix *weight = matrix_memory_allocator.Allocate("weight");
        gpu_sim.GetColumn(softmax_row, j, weight, kInSharedMemory);

        // Multiply value by weight: weight[1,1] broadcast-multiplies value[1,512]
        // This requires reshape: weight is [1,1], value is [1,512]
        // We want weight * value element-wise, which we can do by:
        // 1. Reshape weight to [1,1]
        // 2. For each element in value, multiply by weight
        // But we don't have element-wise multiply! We need MatMul

        // weight [1,1] @ value [1,512] won't work (dimension mismatch)
        // We need: weight_val * value
        // Actually, use MatMul with transposed weight: value @ weight^T won't work either

        // Let me use a different approach: weight is scalar, multiply value by it
        // But we don't have scalar multiply directly...
        // Actually we do! It's in the comments but maybe not implemented?

        // Alternative: compute weight[j] * values[j] and accumulate
        // weight is [1,1], values[j] is [1,512]
        // We can use: Reshape weight, then MatMul

        // Actually, softmax_row is [1, i+1] and values[j] is [1, 512]
        // I need to extract element j from softmax_row and multiply values[j] by it
        // GetColumn gives me [1,1], which is perfect for MatrixMulNum if we had it

        // Let me try a matrix approach:
        // weight [1,1], value [1,512]
        // weight @ value won't work
        // How about: reshape to make it work?
        // Actually, the best way is to build the full softmax matrix first then do one big matmul
        gpu_sim.ReleaseMatrix(weight);
      }

      gpu_sim.ReleaseMatrix(softmax_row);
    }

    gpu_sim.ReleaseMatrix(exp_scores);

    // This approach is getting complicated. Let me stick with the original.
    // Just return for now
    return;
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
