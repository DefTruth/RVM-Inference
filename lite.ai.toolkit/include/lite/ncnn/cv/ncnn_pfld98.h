//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_PFLD98_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_PFLD98_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNPFLD98 : public BasicNCNNHandler
  {
  public:
    explicit NCNNPFLD98(const std::string &_param_path,
                        const std::string &_bin_path,
                        unsigned int _num_threads = 1);

    ~NCNNPFLD98() override = default;

  private:
    const int input_height = 112;
    const int input_width = 112;
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.0f / 255.f, 1.0f / 255.f, 1.0f / 255.f};

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_PFLD98_H
