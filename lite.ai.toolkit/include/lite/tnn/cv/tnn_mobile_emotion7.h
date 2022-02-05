//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_MOBILE_EMOTION7_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_MOBILE_EMOTION7_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNMobileEmotion7 : public BasicTNNHandler
  {
  public:
    explicit TNNMobileEmotion7(const std::string &_proto_path,
                               const std::string &_model_path,
                               unsigned int _num_threads = 1); //
    ~TNNMobileEmotion7() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f, 1.f, 1.f};
    std::vector<float> bias_vals = {-103.939f, -116.779f, -123.68f};
    const char *emotion_texts[7] = {
        "angry", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
    };

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_MOBILE_EMOTION7_H
