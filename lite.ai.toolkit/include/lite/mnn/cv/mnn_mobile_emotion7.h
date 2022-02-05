//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_EMOTION7_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_EMOTION7_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMobileEmotion7 : public BasicMNNHandler
  {
  public:
    explicit MNNMobileEmotion7(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNMobileEmotion7() override = default;

  private:
    const float mean_vals[3] = {103.939f, 116.779f, 123.68f};
    const float norm_vals[3] = {1.f, 1.f, 1.f};
    const char *emotion_texts[7] = {
        "angry", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
    };

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; // padding & resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_EMOTION7_H
