//
// Created by DefTruth on 2021/4/2.
//

#ifndef LITE_AI_ORT_CV_AGE_GOOGLENET_H
#define LITE_AI_ORT_CV_AGE_GOOGLENET_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS AgeGoogleNet : public BasicOrtHandler
  {
  private:
    const float mean_val[3] = {104.0f, 117.0f, 123.0f};
    const float scale_val[3] = {1.0f, 1.0f, 1.0f};
    const unsigned int age_intervals[8][2] = {
        {0,  2},
        {4,  6},
        {8,  12},
        {15, 20},
        {25, 32},
        {38, 43},
        {48, 53},
        {60, 100}
    };

  public:
    explicit AgeGoogleNet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~AgeGoogleNet() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Age &age);
  };

}

#endif //LITE_AI_ORT_CV_AGE_GOOGLENET_H
