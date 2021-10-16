//
// Created by DefTruth on 2021/7/24.
//

#ifndef LITE_AI_ORT_CV_REXNET_EMOTION7_H
#define LITE_AI_ORT_CV_REXNET_EMOTION7_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS ReXNetEmotion7 : public BasicOrtHandler
  {
  public:
    explicit ReXNetEmotion7(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~ReXNetEmotion7() override = default;

  private:
    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float scale_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};

    const char *emotion_texts[7] = {
        "angry", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
    };

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}

#endif //LITE_AI_ORT_CV_REXNET_EMOTION7_H
