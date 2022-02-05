//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_COLORIZER_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_COLORIZER_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNColorizer : public BasicNCNNHandler
  {
  public:
    explicit NCNNColorizer(const std::string &_param_path,
                           const std::string &_bin_path,
                           unsigned int _num_threads = 1); //
    ~NCNNColorizer() override = default;

  private:
    int input_height = 256;
    int input_width = 256;

  private:

    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::ColorizeContent &colorize_content);

  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_COLORIZER_H
