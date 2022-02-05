//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_AGE_GOOGLENET_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_AGE_GOOGLENET_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNAgeGoogleNet : public BasicNCNNHandler
  {
  public:
    explicit NCNNAgeGoogleNet(const std::string &_param_path,
                              const std::string &_bin_path,
                              unsigned int _num_threads = 1);

    ~NCNNAgeGoogleNet() override = default;

  private:
    const int input_height = 224;
    const int input_width = 224;
    const float mean_vals[3] = {104.0f, 117.0f, 123.0f};
    const float norm_vals[3] = {1.0f, 1.0f, 1.0f};

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

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Age &age);
  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_AGE_GOOGLENET_H
