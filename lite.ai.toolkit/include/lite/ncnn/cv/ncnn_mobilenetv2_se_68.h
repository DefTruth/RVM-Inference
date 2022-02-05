//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILENETV2_SE_68_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILENETV2_SE_68_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNMobileNetV2SE68 : public BasicNCNNHandler
  {
  public:
    explicit NCNNMobileNetV2SE68(const std::string &_param_path,
                                 const std::string &_bin_path,
                                 unsigned int _num_threads = 1);

    ~NCNNMobileNetV2SE68() override = default;

  private:
    const int input_height = 56;
    const int input_width = 56;
    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float norm_vals[3] = {1.0f / (255.f * 0.229f), 1.0f / (255.f * 0.224f), 1.0f / (255.f * 0.225f)};

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILENETV2_SE_68_H
