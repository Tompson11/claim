
#pragma once
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

torch::Tensor LidarDepthGenerationCUDA(const torch::Tensor &uvd,
                                       const int point_nums,
                                       const int image_batch,
                                       const int image_height,
                                       const int image_width,
                                       const bool inv_depth = true);

torch::Tensor LidarIntensityGenerationCUDA(const torch::Tensor &uvd,
                                           const torch::Tensor &intensity,
                                           const int point_nums,
                                           const int image_batch,
                                           const int image_height,
                                           const int image_width);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
LidarAndImageIntensityHistogramGenerationCUDA(
    const torch::Tensor &lidar_intensity, const torch::Tensor &image_intensity,
    const int image_batch, const int image_height, const int image_width,
    const int bin_nums);

torch::Tensor LossCalculationCUDA(const torch::Tensor &depth_lidar,
                                  const torch::Tensor &depth_mono,
                                  const int image_batch, const int image_height,
                                  const int image_width, const int shift,
                                  const int box_p);

torch::Tensor LossCalculationCUDA2(const torch::Tensor &depth_lidar,
                                   const torch::Tensor &depth_mono,
                                   const int image_batch,
                                   const int image_height,
                                   const int image_width, const int shift,
                                   const int box_p);