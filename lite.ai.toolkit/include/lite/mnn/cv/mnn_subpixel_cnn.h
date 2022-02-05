//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_SUBPIXEL_CNN_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_SUBPIXEL_CNN_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNSubPixelCNN : public BasicMNNHandler
  {
  public:
    explicit MNNSubPixelCNN(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNSubPixelCNN() override = default;

  private:
    void initialize_pretreat(); // no use

    void transform(const cv::Mat &mat) override; // resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_SUBPIXEL_CNN_H
