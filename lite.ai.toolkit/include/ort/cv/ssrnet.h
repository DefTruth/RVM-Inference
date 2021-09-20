//
// Created by DefTruth on 2021/3/14.
//

#ifndef LITE_AI_ORT_CV_SSRNET_H
#define LITE_AI_ORT_CV_SSRNET_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS SSRNet : public BasicOrtHandler
  {
  private:
    const float mean_val[3] = {0.485f, 0.456f, 0.406f};
    const float scale_val[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};

  public:
    explicit SSRNet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~SSRNet() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Age &age);
  };
}

#endif //LITE_AI_ORT_CV_SSRNET_H
