//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_GENDER_GOOGLENET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_GENDER_GOOGLENET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNGenderGoogleNet : public BasicMNNHandler
  {
  public:
    explicit MNNGenderGoogleNet(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNGenderGoogleNet() override = default;

  private:
    const float mean_vals[3] = {104.0f, 117.0f, 123.0f};
    const float norm_vals[3] = {1.0f, 1.0f, 1.0f};
    const char *gender_texts[2] = {"female", "male"};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::Gender &gender);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_GENDER_GOOGLENET_H
