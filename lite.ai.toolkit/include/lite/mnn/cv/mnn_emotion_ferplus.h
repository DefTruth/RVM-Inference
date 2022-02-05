//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_EMOTION_FERPLUS_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_EMOTION_FERPLUS_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNEmotionFerPlus : public BasicMNNHandler
  {
  public:
    explicit MNNEmotionFerPlus(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNEmotionFerPlus() override = default;

  private:
    const float mean_vals[3] = {0.0f};
    const float norm_vals[3] = {1.0f};
    const char *emotion_texts[8] = {
        "neutral", "happiness", "surprise", "sadness", "anger",
        "disgust", "fear", "contempt"
    };

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; // padding & resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_EMOTION_FERPLUS_H
