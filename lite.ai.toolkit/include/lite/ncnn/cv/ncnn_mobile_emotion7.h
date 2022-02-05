//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILE_EMOTION7_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILE_EMOTION7_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNMobileEmotion7 : public BasicNCNNHandler
  {
  public:
    explicit NCNNMobileEmotion7(const std::string &_param_path,
                                const std::string &_bin_path,
                                unsigned int _num_threads = 1);

    ~NCNNMobileEmotion7() override = default;

  private:
    const int input_height = 224;
    const int input_width = 224;
    const float mean_vals[3] = {103.939f, 116.779f, 123.68f};
    const float norm_vals[3] = {1.f, 1.f, 1.f};
    const char *emotion_texts[7] = {
        "angry", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
    };

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_MOBILE_EMOTION7_H
