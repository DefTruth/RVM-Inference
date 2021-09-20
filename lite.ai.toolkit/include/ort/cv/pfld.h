//
// Created by DefTruth on 2021/3/14.
//

#ifndef LITE_AI_ORT_CV_PFLD_H
#define LITE_AI_ORT_CV_PFLD_H

#include "ort/core/ort_core.h"

namespace ortcv
{

  class LITE_EXPORTS PFLD : public BasicOrtHandler
  {
  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0f / 255.0f;

  public:
    explicit PFLD(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~PFLD() override = default; // override

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);

  };
}

#endif //LITE_AI_ORT_CV_PFLD_H
