//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_MOBILENETV2_SE_68_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_MOBILENETV2_SE_68_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNMobileNetV2SE68 : public BasicTNNHandler
  {
  public:
    explicit TNNMobileNetV2SE68(const std::string &_proto_path,
                                const std::string &_model_path,
                                unsigned int _num_threads = 1); //
    ~TNNMobileNetV2SE68() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {
        1.0f / (255.f * 0.229f),
        1.0f / (255.f * 0.224f),
        1.0f / (255.f * 0.225f)
    };
    std::vector<float> bias_vals = {
        -255.f * 0.485f * (1.0f / (255.f * 0.229f)),
        -255.f * 0.456f * (1.0f / (255.f * 0.224f)),
        -255.f * 0.406f * (1.0f / (255.f * 0.225f))
    };

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}


#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_MOBILENETV2_SE_68_H
