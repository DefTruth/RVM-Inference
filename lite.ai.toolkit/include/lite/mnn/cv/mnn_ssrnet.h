//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_SSRNET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_SSRNET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNSSRNet : public BasicMNNHandler
  {
  public:
    explicit MNNSSRNet(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNSSRNet() override = default;

  private:
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {(1.0f / 0.229f) * (1.0f / 255.0f),
                                (1.0f / 0.224f) * (1.0f / 255.0f),
                                (1.0f / 0.225f) * (1.0f / 255.0f)};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; // padding & resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::Age &age);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_SSRNET_H
