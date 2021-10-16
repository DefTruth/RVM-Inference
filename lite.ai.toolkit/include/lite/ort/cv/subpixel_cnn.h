//
// Created by DefTruth on 2021/4/5.
//

#ifndef LITE_AI_ORT_CV_SUBPIXEL_CNN_H
#define LITE_AI_ORT_CV_SUBPIXEL_CNN_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS SubPixelCNN : public BasicOrtHandler
  {
  public:
    explicit SubPixelCNN(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~SubPixelCNN() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);
  };
}

#endif //LITE_AI_ORT_CV_SUBPIXEL_CNN_H
