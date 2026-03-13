#include "simulator.hpp"
#include <iostream>

int main() {
    sjtu::GpuSimulator gpu_sim;
    sjtu::MatrixMemoryAllocator alloc;

    // Create a simple matrix in HBM
    std::vector<float> data(512, 1.0f);
    sjtu::Matrix *m1 = new sjtu::Matrix(1, 512, data, gpu_sim);
    alloc.Bind(m1, "m1");

    std::cerr << "Initial position: " << m1->GetPosition() << std::endl;

    // Try to move it to SRAM
    gpu_sim.MoveMatrixToSharedMem(m1);

    std::cerr << "After queuing move, position: " << m1->GetPosition() << std::endl;

    // Run simulator
    gpu_sim.Run(false, &alloc);

    std::cerr << "After run, position: " << m1->GetPosition() << std::endl;

    return 0;
}
