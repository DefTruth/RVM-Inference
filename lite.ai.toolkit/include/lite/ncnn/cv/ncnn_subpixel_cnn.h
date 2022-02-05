//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_SUBPIXEL_CNN_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_SUBPIXEL_CNN_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNSubPixelCNN : public BasicNCNNHandler
  {
  public:
    explicit NCNNSubPixelCNN(const std::string &_param_path,
                             const std::string &_bin_path,
                             unsigned int _num_threads = 1); //
    ~NCNNSubPixelCNN() override = default;

  private:
    int input_height = 224;
    int input_width = 224;

  private:

    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);

  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_SUBPIXEL_CNN_H
