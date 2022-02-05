//
// Created by DefTruth on 2021/11/27.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_GENDER_GOOGLENET_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_GENDER_GOOGLENET_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNGenderGoogleNet : public BasicTNNHandler
  {
  public:
    explicit TNNGenderGoogleNet(const std::string &_proto_path,
                                const std::string &_model_path,
                                unsigned int _num_threads = 1); //
    ~TNNGenderGoogleNet() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.0f, 1.0f, 1.0f};
    std::vector<float> bias_vals = {-104.0f, -117.0f, -123.0f};
    const char *gender_texts[2] = {"female", "male"};

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Gender &gender);
  };
}


#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_GENDER_GOOGLENET_H
