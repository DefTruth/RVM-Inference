//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_EMOTION_FERPLUS_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_EMOTION_FERPLUS_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNEmotionFerPlus : public BasicNCNNHandler
  {
  public:
    explicit NCNNEmotionFerPlus(const std::string &_param_path,
                                const std::string &_bin_path,
                                unsigned int _num_threads = 1);

    ~NCNNEmotionFerPlus() override = default;

  private:
    const int input_height = 64;
    const int input_width = 64;
    const float mean_vals[1] = {0.f};
    const float norm_vals[1] = {1.0f};
    const char *emotion_texts[8] = {
        "neutral", "happiness", "surprise", "sadness", "anger",
        "disgust", "fear", "contempt"
    };

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_EMOTION_FERPLUS_H
