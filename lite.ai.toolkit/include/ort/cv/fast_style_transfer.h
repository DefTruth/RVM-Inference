//
// Created by DefTruth on 2021/4/3.
//

#ifndef LITE_AI_ORT_CV_FAST_STYLE_TRANSFER_H
#define LITE_AI_ORT_CV_FAST_STYLE_TRANSFER_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS FastStyleTransfer : public BasicOrtHandler
  {
  public:
    explicit FastStyleTransfer(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~FastStyleTransfer() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::StyleContent &style_content);
  };
}
#endif //LITE_AI_ORT_CV_FAST_STYLE_TRANSFER_H
