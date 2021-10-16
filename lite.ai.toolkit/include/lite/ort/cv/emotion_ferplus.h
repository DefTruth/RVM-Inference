//
// Created by DefTruth on 2021/4/3.
//

#ifndef LITE_AI_ORT_CV_EMOTION_FERPLUS_H
#define LITE_AI_ORT_CV_EMOTION_FERPLUS_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS EmotionFerPlus : public BasicOrtHandler
  {
  private:
    const char *emotion_texts[8] = {
        "netural", "happiness", "surprise", "sadness", "anger",
        "disgust", "fear", "contempt"
    };
  public:
    explicit EmotionFerPlus(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~EmotionFerPlus() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}

#endif //LITE_AI_ORT_CV_EMOTION_FERPLUS_H
