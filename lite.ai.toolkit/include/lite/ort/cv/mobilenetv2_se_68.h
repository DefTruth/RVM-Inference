//
// Created by DefTruth on 2021/7/27.
//

#ifndef LITE_AI_ORT_CV_MOBILENETV2_SE_68_H
#define LITE_AI_ORT_CV_MOBILENETV2_SE_68_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS MobileNetV2SE68 : public BasicOrtHandler
  {
  public:
    explicit MobileNetV2SE68(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~MobileNetV2SE68() override = default; // override

  private:
    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float scale_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);

  };
}


#endif //LITE_AI_ORT_CV_MOBILENETV2_SE_68_H
