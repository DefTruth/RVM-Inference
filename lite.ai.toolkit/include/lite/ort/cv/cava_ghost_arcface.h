//
// Created by DefTruth on 2021/7/18.
//

#ifndef LITE_AI_ORT_CV_CAVA_GHOST_ARCFACE_H
#define LITE_AI_ORT_CV_CAVA_GHOST_ARCFACE_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS CavaGhostArcFace : public BasicOrtHandler
  {
  public:
    explicit CavaGhostArcFace(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~CavaGhostArcFace() override = default;

  private:
    static constexpr const float mean_val = 127.5f;
    static constexpr const float scale_val = 1.f / 128.0f;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content);
  };
}

#endif //LITE_AI_ORT_CV_CAVA_GHOST_ARCFACE_H
