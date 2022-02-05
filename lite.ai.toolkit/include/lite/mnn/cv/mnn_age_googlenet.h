//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_AGE_GOOGLENET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_AGE_GOOGLENET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNAgeGoogleNet : public BasicMNNHandler
  {
  public:
    explicit MNNAgeGoogleNet(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNAgeGoogleNet() override = default;

  private:
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
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; // padding & resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::Age &age);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_AGE_GOOGLENET_H
