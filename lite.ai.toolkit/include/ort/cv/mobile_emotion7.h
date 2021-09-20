//
// Created by DefTruth on 2021/7/24.
//

#ifndef LITE_AI_ORT_CV_MOBILE_EMOTION7_H
#define LITE_AI_ORT_CV_MOBILE_EMOTION7_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS MobileEmotion7 : public BasicOrtHandler
{
  public:
  explicit MobileEmotion7(const std::string &_onnx_path, unsigned int _num_threads = 1) :
      BasicOrtHandler(_onnx_path, _num_threads)
  {};

  ~MobileEmotion7() override = default;

  private:
    const float mean_vals[3] = {103.939f, 116.779f, 123.68f};
    const float scale_vals[3] = {1.f, 1.f, 1.f};

    const char *emotion_texts[7] = {
      "angry", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
  };

  private:
  Ort::Value transform(const cv::Mat &mat) override;

  public:
  void detect(const cv::Mat &mat, types::Emotions &emotions);
};
}


#endif //LITE_AI_ORT_CV_MOBILE_EMOTION7_H
