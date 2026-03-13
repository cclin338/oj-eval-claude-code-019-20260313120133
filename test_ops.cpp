#include "simulator.hpp"
#include <iostream>

int main() {
    sjtu::GpuSimulator gpu_sim;
    sjtu::MatrixMemoryAllocator alloc;

    // Create matrices
    std::vector<float> data1(512, 1.0f);
    std::vector<float> data2(512, 2.0f);

    sjtu::Matrix *m1 = new sjtu::Matrix(1, 512, data1, gpu_sim);
    sjtu::Matrix *m2 = new sjtu::Matrix(1, 512, data2, gpu_sim);
    alloc.Bind(m1, "m1");
    alloc.Bind(m2, "m2");

    std::cerr << "Initial: m1=" << m1->GetPosition() << " m2=" << m2->GetPosition() << std::endl;

    // Move both to SRAM
    gpu_sim.MoveMatrixToSharedMem(m1);
    gpu_sim.MoveMatrixToSharedMem(m2);

    std::cerr << "After queue: m1=" << m1->GetPosition() << " m2=" << m2->GetPosition() << std::endl;

    // Try to add them
    sjtu::Matrix *result = alloc.Allocate("result");
    gpu_sim.MatAdd(m1, m2, result);

    std::cerr << "After queue matAdd: m1=" << m1->GetPosition() << " m2=" << m2->GetPosition()
              << " result=" << result->GetPosition() << std::endl;

    // Run simulator
    std::cerr << "Running simulator..." << std::endl;
    gpu_sim.Run(false, &alloc);

    std::cerr << "After run: m1=" << m1->GetPosition() << " m2=" << m2->GetPosition()
              << " result=" << result->GetPosition() << std::endl;

    return 0;
}
