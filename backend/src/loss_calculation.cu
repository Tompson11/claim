#include "loss_calculation.h"
#include <chrono>
#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <torch/extension.h>
#include <tuple>
namespace cg = cooperative_groups;

__inline__ __device__ const float get_voxel_value(const float *data,
                                                  const unsigned int u,
                                                  const unsigned int v,
                                                  const unsigned int h) {
  return data[u * h + v];
}

__global__ void depth_sim_loss_kernel(
    const float *depth_lidar, const float *depth_mono, const unsigned int B,
    const unsigned int H, const unsigned int W, const unsigned int box_p,
    const unsigned int num_box_h, const unsigned int num_box_w,
    const unsigned int shift, float *out_loss) {
  unsigned int batch_id = threadIdx.x;
  if (batch_id >= B)
    return;

  unsigned long long offset = batch_id * H * W;
  const float *src = depth_lidar + offset;
  const float *tgt = depth_mono;

  unsigned int patch_id_x = blockIdx.x;
  unsigned int patch_id_y = blockIdx.y;
  unsigned int u0 = patch_id_x * box_p + shift;
  unsigned int v0 = patch_id_y * box_p + shift;
  unsigned int u1 = u0 + box_p;
  unsigned int v1 = v0 + box_p;
  float uc = (u0 + u1) / 2.0;
  float vc = (v0 + v1) / 2.0;

  unsigned int cnt = 0;
  float x = 0, y = 0;
  float x2 = 0, y2 = 0;
  float xy = 0;
  float u_sum = 0, v_sum = 0;
  for (unsigned int u = u0; u < u1; ++u)
    for (unsigned int v = v0; v < v1; ++v) {
      const float dx = src[v * W + u];
      if (dx > 0.0) {
        ++cnt;
        u_sum += (u - uc);
        v_sum += (v - vc);

        const float dy = tgt[v * W + u];

        x += dx;
        y += dy;
        x2 += dx * dx;
        y2 += dy * dy;
        xy += dx * dy;
      }
    }

  unsigned long long num_box = num_box_h * num_box_w;
  unsigned long long out_offset =
      batch_id * num_box + patch_id_x * num_box_h + patch_id_y;
  if (cnt < 15 || (fabs(u_sum / cnt) > 0.25 * box_p) ||
      (fabs(v_sum / cnt) > 0.25 * box_p))
    // if (cnt < 5)
    out_loss[out_offset] = 0.0;
  else {
    float coef = ((xy * cnt) - (x * y)) / sqrt(x2 * cnt - x * x) /
                 sqrt(y2 * cnt - y * y);
    
    if (isnan(coef))
      out_loss[out_offset] = 1.0;
    else if (isinf(coef))
      out_loss[out_offset] = 1.0;
    else if (coef > 1)
      out_loss[out_offset] = 1e-4;
    else
      out_loss[out_offset] = 1.0 - coef;
  }
}

__global__ void depth_sim_loss_kernel2(
    const float *depth_lidar, const float *depth_mono, const unsigned int B,
    const unsigned int H, const unsigned int W, const unsigned int box_p,
    const unsigned int num_box_h, const unsigned int num_box_w,
    const unsigned int shift, float *out_loss) {
  unsigned int batch_id = threadIdx.x;
  if (batch_id >= B)
    return;

  unsigned long long offset = batch_id * H * W;
  const float *src = depth_lidar + offset;
  const float *tgt = depth_mono;

  unsigned int patch_id_x = blockIdx.x;
  unsigned int patch_id_y = blockIdx.y;
  unsigned int u0 = patch_id_x * box_p + shift;
  unsigned int v0 = patch_id_y * box_p + shift;
  unsigned int u1 = u0 + box_p;
  unsigned int v1 = v0 + box_p;

  unsigned int cnt = 0;
  float last_dx = -1.0;
  float last_dy = -1.0;
  float score1 = 0.0;
  for (unsigned int u = u0; u < u1; ++u)
    for (unsigned int v = v0; v < v1; ++v) {
      const float dx = src[v * W + u];
      if (dx > 0.0) {
        const float dy = tgt[v * W + u];
        if (cnt > 0) {
          if ((dx > last_dx) ^ (dy > last_dy)) {

          } else {
            ++score1;
          }
        }
        last_dx = dx;
        last_dy = dy;
        ++cnt;
      }
    }

  cnt = 0;
  float score2 = 0.0;
  for (unsigned int v = v0; v < v1; ++v)
    for (unsigned int u = u0; u < u1; ++u) {
      const float dx = src[v * W + u];
      if (dx > 0.0) {
        const float dy = tgt[v * W + u];
        if (cnt > 0) {
          if ((dx > last_dx) ^ (dy > last_dy)) {

          } else {
            ++score2;
          }
        }
        last_dx = dx;
        last_dy = dy;
        ++cnt;
      }
    }

  unsigned long long num_box = num_box_h * num_box_w;
  unsigned long long out_offset =
      batch_id * num_box + patch_id_x * num_box_h + patch_id_y;
  if (cnt > 5)
    out_loss[out_offset] =
        1.0 - (score1 / (cnt - 1) + score2 / (cnt - 1)) / 2.0;
  else
    out_loss[out_offset] = 0.0;
}

__global__ void lidar_depth_kernel(const float *uvd, const int P, const int B,
                                   const int H, const int W, float *out_depth, 
                                   const bool inv_depth) {
  unsigned int batch_id = threadIdx.x;
  unsigned int point_id = blockIdx.x;
  unsigned long long uvd_offset = batch_id * 3 * P;
  unsigned long long p_offset = point_id * 3;

  const float *uvd_cur = uvd + uvd_offset + p_offset;

  if (uvd_cur[2] < 0.02)
    return;

  int u = uvd_cur[0];
  if (u < 0 || u >= W)
    return;

  int v = uvd_cur[1];
  if (v < 0 || v >= H)
    return;

  unsigned long long out_offset = batch_id * H * W + v * W + u;
  if(inv_depth)
    out_depth[out_offset] = 1.0 / uvd_cur[2];
  else
    out_depth[out_offset] = uvd_cur[2];
}

__global__ void lidar_intensity_kernel(const float *uvd, const float *intensity,
                                       const int P, const int B, const int H,
                                       const int W, float *out_intensity) {
  unsigned int batch_id = threadIdx.x;
  unsigned int point_id = blockIdx.x;
  unsigned long long uvd_offset = batch_id * 3 * P;
  unsigned long long p_offset = point_id * 3;

  const float *uvd_cur = uvd + uvd_offset + p_offset;
  // printf("%d-%d-%f-%f-%f\n", int(batch_id), int(point_id), uvd_cur[0],
  // uvd_cur[1], uvd_cur[2]);

  if (uvd_cur[2] < 0.02)
    return;

  int u = uvd_cur[0];
  if (u < 0 || u >= W)
    return;

  int v = uvd_cur[1];
  if (v < 0 || v >= H)
    return;

  unsigned long long out_offset = batch_id * H * W + v * W + u;
  out_intensity[out_offset] = intensity[point_id];

  // int radius = 1;
  // for (int i = max(0, v - radius); i <= min(H - 1, v + radius); ++i)
  //   for (int j = max(0, u - radius); j <= min(W - 1, u + radius); ++j) {
  //     unsigned long long out_offset = batch_id * H * W + i * W + j;
  //     out_intensity[out_offset] = intensity[point_id];
  //   }
}

__global__ void intensity_hist_kernel(const float *lidar_intensity,
                                      const float *image_intensity, const int B,
                                      const int H, const int W, const int M,
                                      float *lidar_hist, float *image_hist,
                                      float *mutual_hist) {
  unsigned int batch_id = threadIdx.x;
  unsigned int col_id = blockIdx.x;

  unsigned long long offset = batch_id * H * W;
  const float *lid_int = lidar_intensity + offset;
  const float *img_int = image_intensity;

  float *lid_hist = lidar_hist + batch_id * W * M;
  float *img_hist = image_hist + batch_id * W * M;
  float *mut_hist = mutual_hist + batch_id * W * M * M;

  for (int i = 0; i < H; ++i) {
    int index = i * W + col_id;
    if (lid_int[index] > 0.0) {
      int lid_bin = min(int(lid_int[index] * M), M - 1);
      int img_bin = min(int(img_int[index] * M), M - 1);
      int mult_bin = lid_bin * M + img_bin;
      ++lid_hist[col_id * M + lid_bin];
      ++img_hist[col_id * M + img_bin];
      ++mut_hist[col_id * M * M + mult_bin];
    }
  }
}

torch::Tensor LidarDepthGenerationCUDA(const torch::Tensor &uvd,
                                       const int point_nums,
                                       const int image_batch,
                                       const int image_height,
                                       const int image_width,
                                       const bool inv_depth) {
  const int N = point_nums;
  const int B = image_batch;
  const int H = image_height;
  const int W = image_width;
  // printf("%d %d %d %d", N, B, H, W);

  auto float_opts = uvd.options().dtype(torch::kFloat32);
  torch::Tensor out_depth = torch::full({B, H, W}, 0.0, float_opts);

  lidar_depth_kernel<<<N, B>>>(uvd.contiguous().data<float>(), N, B, H, W,
                               out_depth.contiguous().data<float>(), inv_depth);

  return out_depth;
}

torch::Tensor LidarIntensityGenerationCUDA(const torch::Tensor &uvd,
                                           const torch::Tensor &intensity,
                                           const int point_nums,
                                           const int image_batch,
                                           const int image_height,
                                           const int image_width) {
  const int N = point_nums;
  const int B = image_batch;
  const int H = image_height;
  const int W = image_width;
  // printf("%d %d %d %d", N, B, H, W);

  auto float_opts = uvd.options().dtype(torch::kFloat32);
  torch::Tensor out_intensity = torch::full({B, H, W}, 0.0, float_opts);

  lidar_intensity_kernel<<<N, B>>>(
      uvd.contiguous().data<float>(), intensity.contiguous().data<float>(), N,
      B, H, W, out_intensity.contiguous().data<float>());

  return out_intensity;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
LidarAndImageIntensityHistogramGenerationCUDA(
    const torch::Tensor &lidar_intensity, const torch::Tensor &image_intensity,
    const int image_batch, const int image_height, const int image_width,
    const int bin_nums) {
  const int B = image_batch;
  const int H = image_height;
  const int W = image_width;
  const int M = bin_nums;
  // printf("%d %d %d %d", N, B, H, W);

  auto float_opts = lidar_intensity.options().dtype(torch::kFloat32);
  // calculate per column
  torch::Tensor lidar_hist = torch::full({B, W, M}, 0.0, float_opts);
  torch::Tensor image_hist = torch::full({B, W, M}, 0.0, float_opts);
  torch::Tensor mutual_hist = torch::full({B, W, M * M}, 0.0, float_opts);

  intensity_hist_kernel<<<W, B>>>(lidar_intensity.contiguous().data<float>(),
                                  image_intensity.contiguous().data<float>(), B,
                                  H, W, M,
                                  lidar_hist.contiguous().data<float>(),
                                  image_hist.contiguous().data<float>(),
                                  mutual_hist.contiguous().data<float>());

  return std::tie(lidar_hist, image_hist, mutual_hist);
}

torch::Tensor LossCalculationCUDA(const torch::Tensor &depth_lidar,
                                  const torch::Tensor &depth_mono,
                                  const int image_batch, const int image_height,
                                  const int image_width, const int shift,
                                  const int box_p) {
  const int B = image_batch;
  const int H = image_height;
  const int W = image_width;
  int num_box_h = (H - shift) / box_p;
  int num_box_w = (W - shift) / box_p;
  const int N = num_box_h * num_box_w;

  auto float_opts = depth_lidar.options().dtype(torch::kFloat32);

  torch::Tensor out_loss = torch::full({B, N}, 0.0, float_opts);

  dim3 blocksize(num_box_w, num_box_h);
  depth_sim_loss_kernel<<<blocksize, B>>>(
      depth_lidar.contiguous().data<float>(),
      depth_mono.contiguous().data<float>(), B, H, W, box_p, num_box_h,
      num_box_w, shift, out_loss.contiguous().data<float>());

  return out_loss;
}

torch::Tensor LossCalculationCUDA2(const torch::Tensor &depth_lidar,
                                   const torch::Tensor &depth_mono,
                                   const int image_batch,
                                   const int image_height,
                                   const int image_width, const int shift,
                                   const int box_p) {
  const int B = image_batch;
  const int H = image_height;
  const int W = image_width;
  int num_box_h = (H - shift) / box_p;
  int num_box_w = (W - shift) / box_p;
  const int N = num_box_h * num_box_w;

  auto float_opts = depth_lidar.options().dtype(torch::kFloat32);

  torch::Tensor out_loss = torch::full({B, N}, 0.0, float_opts);

  dim3 blocksize(num_box_w, num_box_h);
  depth_sim_loss_kernel2<<<blocksize, B>>>(
      depth_lidar.contiguous().data<float>(),
      depth_mono.contiguous().data<float>(), B, H, W, box_p, num_box_h,
      num_box_w, shift, out_loss.contiguous().data<float>());

  return out_loss;
}