//
// Created by DefTruth on 2021/4/9.
//

#ifndef LITE_AI_ORT_CV_COLORIZER_H
#define LITE_AI_ORT_CV_COLORIZER_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS Colorizer : public BasicOrtHandler
  {
  public:
    explicit Colorizer(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~Colorizer() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::ColorizeContent &colorize_content);
  };
}

#endif //LITE_AI_ORT_CV_COLORIZER_H
