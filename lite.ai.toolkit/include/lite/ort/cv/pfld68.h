//
// Created by DefTruth on 2021/7/27.
//

#ifndef LITE_AI_ORT_CV_PFLD68_H
#define LITE_AI_ORT_CV_PFLD68_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS PFLD68 : public BasicOrtHandler
  {

  public:
    explicit PFLD68(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~PFLD68() override = default; // override

  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0f / 255.0f;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);

  };
}

#endif //LITE_AI_ORT_CV_PFLD68_H
