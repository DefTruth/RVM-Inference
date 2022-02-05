//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_EFFICIENT_EMOTION8_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_EFFICIENT_EMOTION8_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNEfficientEmotion8 : public BasicNCNNHandler
  {
  public:
    explicit NCNNEfficientEmotion8(const std::string &_param_path,
                                   const std::string &_bin_path,
                                   unsigned int _num_threads = 1);

    ~NCNNEfficientEmotion8() override = default;

  private:
    const int input_height = 224;
    const int input_width = 224;
    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float norm_vals[3] = {1.f / (255.f * 0.229f), 1.f / (255.f * 0.224f), 1.f / (255.f * 0.225f)};
    const char *emotion_texts[8] = {
        "angry", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
    };

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_EFFICIENT_EMOTION8_H
