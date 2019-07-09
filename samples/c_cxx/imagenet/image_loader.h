// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <string>
#include "cached_interpolation.h"
#include "sync_api.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>

template <typename T>
void ResizeImageInMemory(const T* input_data, float* output_data, int in_height, int in_width, int out_height,
                         int out_width, int channels);

/**
 * CalculateResizeScale determines the float scaling factor.
 * @param in_size
 * @param out_size
 * @param align_corners If true, the centers of the 4 corner pixels of the input and output tensors are aligned,
 *                        preserving the values at the corner pixels
 * @return
 */
inline float CalculateResizeScale(int64_t in_size, int64_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

class RunnableTask : public std::unary_function<void, void> {
 public:
  virtual void operator()(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci) noexcept = 0;
  virtual ~RunnableTask() = default;
};

class DataProcessing {
 public:
  virtual void operator()(const void* input_data, void* output_data) = 0;
  virtual size_t GetOutputSizeInBytes(size_t batch_size) = 0;
  virtual std::vector<int64_t> GetOutputShape(size_t batch_size) = 0;
};

template <typename InputType>
class OutputCollector {
 public:
  virtual void operator()(const std::vector<InputType>& task_id_list, const OrtValue* tensor) = 0;
  // release the internal cache
  virtual void ResetCache() = 0;
};

class InceptionPreprocessing : public DataProcessing {
 private:
  int out_height_;
  int out_width_;
  int channels_;

 public:
  InceptionPreprocessing(int out_height, int out_width, int channels)
      : out_height_(out_height), out_width_(out_width), channels_(channels) {}

  void operator()(const void* input_data, void* output_data) override;
  size_t GetOutputSizeInBytes(size_t batch_size) override {
    return out_height_ * out_width_ * channels_ * batch_size * sizeof(float);
  }
  std::vector<int64_t> GetOutputShape(size_t batch_size) override {
    return {(int64_t)batch_size, out_height_, out_width_, channels_};
  }
};
