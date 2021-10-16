//
// Created by DefTruth on 2021/4/4.
//

#ifndef LITE_AI_ORT_CV_ARCAFCE_RESNET_H
#define LITE_AI_ORT_CV_ARCAFCE_RESNET_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS GlintArcFace : public BasicOrtHandler
  {
  public:
    explicit GlintArcFace(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~GlintArcFace() override = default;

  private:
    static constexpr const float mean_val = 127.5f;
    static constexpr const float scale_val = 1.f / 127.5f;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content);
  };
}

#endif //LITE_AI_ORT_CV_ARCAFCE_RESNET_H
