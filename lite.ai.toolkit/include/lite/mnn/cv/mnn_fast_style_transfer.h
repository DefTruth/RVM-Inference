//
// Created by DefTruth on 2021/11/29.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FAST_STYLE_TRANSFER_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FAST_STYLE_TRANSFER_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFastStyleTransfer : public BasicMNNHandler
  {
  public:
    explicit MNNFastStyleTransfer(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNFastStyleTransfer() override = default;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f, 1.f, 1.f};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; // resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::StyleContent &style_content);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FAST_STYLE_TRANSFER_H
