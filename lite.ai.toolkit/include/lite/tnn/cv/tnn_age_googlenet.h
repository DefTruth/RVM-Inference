//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_AGE_GOOGLENET_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_AGE_GOOGLENET_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNAgeGoogleNet : public BasicTNNHandler
  {
  public:
    explicit TNNAgeGoogleNet(const std::string &_proto_path,
                             const std::string &_model_path,
                             unsigned int _num_threads = 1); //
    ~TNNAgeGoogleNet() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.0f, 1.0f, 1.0f};
    std::vector<float> bias_vals = {-104.0f, -117.0f, -123.0f};
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
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Age &age);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_AGE_GOOGLENET_H
