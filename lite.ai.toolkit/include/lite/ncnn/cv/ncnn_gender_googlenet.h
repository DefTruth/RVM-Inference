//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_GENDER_GOOGLENET_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_GENDER_GOOGLENET_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNGenderGoogleNet : public BasicNCNNHandler
  {
  public:
    explicit NCNNGenderGoogleNet(const std::string &_param_path,
                                 const std::string &_bin_path,
                                 unsigned int _num_threads = 1);

    ~NCNNGenderGoogleNet() override = default;

  private:
    const int input_height = 224;
    const int input_width = 224;
    const float mean_vals[3] = {104.0f, 117.0f, 123.0f};
    const float norm_vals[3] = {1.0f, 1.0f, 1.0f};
    const char *gender_texts[2] = {"female", "male"};

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Gender &gender);
  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_GENDER_GOOGLENET_H
