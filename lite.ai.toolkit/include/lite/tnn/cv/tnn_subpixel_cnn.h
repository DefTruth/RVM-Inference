//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_SUBPIXEL_CNN_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_SUBPIXEL_CNN_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNSubPixelCNN : public BasicTNNHandler
  {
  public:
    explicit TNNSubPixelCNN(const std::string &_proto_path,
                            const std::string &_model_path,
                            unsigned int _num_threads = 1); //
    ~TNNSubPixelCNN() override = default;

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_SUBPIXEL_CNN_H
