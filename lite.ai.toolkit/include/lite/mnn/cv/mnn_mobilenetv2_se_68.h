//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILENETV2_SE_68_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILENETV2_SE_68_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMobileNetV2SE68 : public BasicMNNHandler
  {
  public:
    explicit MNNMobileNetV2SE68(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNMobileNetV2SE68() override = default;

  private:
    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float norm_vals[3] = {1.0f / (255.f * 0.229f), 1.0f / (255.f * 0.224f), 1.0f / (255.f * 0.225f)};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILENETV2_SE_68_H
