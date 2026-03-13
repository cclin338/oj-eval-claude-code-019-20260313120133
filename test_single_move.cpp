#include "simulator.hpp"
#include <iostream>

int main() {
    sjtu::GpuSimulator gpu_sim;
    sjtu::MatrixMemoryAllocator alloc;

    std::vector<float> data(512, 1.0f);
    sjtu::Matrix *m = new sjtu::Matrix(1, 512, data, gpu_sim);
    alloc.Bind(m, "m");

    // Move to SRAM
    gpu_sim.MoveMatrixToSharedMem(m);

    // Use it in a calculation
    sjtu::Matrix *m_copy = alloc.Allocate("m_copy");
    gpu_sim.Copy(m, m_copy, sjtu::kInSharedMemory);

    // Move back to HBM
    gpu_sim.MoveMatrixToGpuHbm(m);

    // Run simulator
    std::cerr << "Running simulator..." << std::endl;
    gpu_sim.Run(false, &alloc);

    std::cerr << "Success! m position=" << m->GetPosition() << std::endl;

    return 0;
}
