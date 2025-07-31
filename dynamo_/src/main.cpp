#include <torch/torch.h>
#include <iostream>

int main(void) {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << "CPU Tensor Preview:" << std::endl;
  std::cout << tensor << std::endl;

  std::cout << "GPU Tensor Preview" << std::endl;
  std::cout << tensor.to(torch::kCUDA);
  return 0;
}
