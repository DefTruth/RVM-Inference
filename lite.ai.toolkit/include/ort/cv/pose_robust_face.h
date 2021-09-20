//
// Created by DefTruth on 2021/7/18.
//

#ifndef LITE_AI_ORT_CV_POSE_ROBUST_FACE_H
#define LITE_AI_ORT_CV_POSE_ROBUST_FACE_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS PoseRobustFace
  {
    // pose robust face: multi-inputs & single output.
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    unsigned int num_outputs = 1;
    unsigned int num_inputs = 1;
    std::vector<float> input_values_handler; // 1x3x224x224
    std::vector<float> yaw_values_handler; // (1,)

  public:
    explicit PoseRobustFace(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~PoseRobustFace();

  public:
    PoseRobustFace(const PoseRobustFace &) = delete;

    PoseRobustFace(PoseRobustFace &&) = delete;

    PoseRobustFace &operator=(const PoseRobustFace &) = delete;

    PoseRobustFace &operator=(PoseRobustFace &&) = delete;

  private:
    const unsigned int num_threads; // init at runtime.

    // normalize to (0., 1.)
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.f / 255.0f;

  private:
    Ort::Value transform(const cv::Mat &mat);

    void print_debug_string();

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content, float yaw = 0.f);
  };
}

#endif //LITE_AI_ORT_CV_POSE_ROBUST_FACE_H
