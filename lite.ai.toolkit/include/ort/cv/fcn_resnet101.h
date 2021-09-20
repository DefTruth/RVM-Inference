//
// Created by DefTruth on 2021/6/14.
//

#ifndef LITE_AI_ORT_CV_FCN_RESNET101_H
#define LITE_AI_ORT_CV_FCN_RESNET101_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS FCNResNet101
  {
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> dynamic_input_node_dims; // >=1 inputs.
    unsigned int dynamic_input_height = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_width = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_tensor_size = 1; // init only, will change according to input mat.
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    unsigned int num_outputs = 1;
    unsigned int num_inputs = 1;
    std::vector<float> dynamic_input_values_handler;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    // single input with dynamic height and width.
    explicit FCNResNet101(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~FCNResNet101();

  protected:
    FCNResNet101(const FCNResNet101 &) = delete;

    FCNResNet101(FCNResNet101 &&) = delete;

    FCNResNet101 &operator=(const FCNResNet101 &) = delete;

    FCNResNet101 &operator=(FCNResNet101 &&) = delete;

  private:
    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    const float scale_vals[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};
    const char *class_names[20] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    }; // 20 classes

  private:
    Ort::Value transform(const cv::Mat &mat);

    void print_debug_string();

  public:
    void detect(const cv::Mat &mat, types::SegmentContent &content);

  };
}

#endif //LITE_AI_ORT_CV_FCN_RESNET101_H
