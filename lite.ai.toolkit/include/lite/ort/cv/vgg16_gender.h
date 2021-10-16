//
// Created by DefTruth on 2021/4/4.
//

#ifndef LITE_AI_ORT_CV_VGG16_GENDER_H
#define LITE_AI_ORT_CV_VGG16_GENDER_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS VGG16Gender : public BasicOrtHandler
  {
  private:
    const char *gender_texts[2] = {"female", "male"};

  public:
    explicit VGG16Gender(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~VGG16Gender() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Gender &gender);
  };
}
#endif //LITE_AI_ORT_CV_VGG16_GENDER_H
