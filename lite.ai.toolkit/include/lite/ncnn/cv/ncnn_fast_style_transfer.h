//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_FAST_STYLE_TRANSFER_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_FAST_STYLE_TRANSFER_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNFastStyleTransfer : public BasicNCNNHandler
  {
  public:
    explicit NCNNFastStyleTransfer(const std::string &_param_path,
                                   const std::string &_bin_path,
                                   unsigned int _num_threads = 1); //
    ~NCNNFastStyleTransfer() override = default;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f, 1.f, 1.f};

  private:
    int input_height = 224;
    int input_width = 224;

  private:

    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::StyleContent &style_content);

  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_FAST_STYLE_TRANSFER_H
