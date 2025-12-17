#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "src/loss_calculation.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_sim_loss", &LossCalculationCUDA);
  m.def("depth_sim_loss2", &LossCalculationCUDA2);
  m.def("generate_lidar_depth", &LidarDepthGenerationCUDA);
  m.def("generate_lidar_intensity", &LidarIntensityGenerationCUDA);
  m.def("generate_histogram", &LidarAndImageIntensityHistogramGenerationCUDA);
}