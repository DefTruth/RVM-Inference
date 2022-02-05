//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_EMOTION_FERPLUS_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_EMOTION_FERPLUS_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNEmotionFerPlus : public BasicTNNHandler
  {
  public:
    explicit TNNEmotionFerPlus(const std::string &_proto_path,
                               const std::string &_model_path,
                               unsigned int _num_threads = 1); //
    ~TNNEmotionFerPlus() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.0f};
    std::vector<float> bias_vals = {0.f};
    const char *emotion_texts[8] = {
        "neutral", "happiness", "surprise", "sadness", "anger",
        "disgust", "fear", "contempt"
    };

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}


#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_EMOTION_FERPLUS_H
