//
// Created by DefTruth on 2021/11/25.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_FSANET_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_FSANET_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNFSANet : public BasicTNNHandler
  {
  public:
    explicit TNNFSANet(const std::string &_proto_path,
                       const std::string &_model_path,
                       unsigned int _num_threads = 1); //
    ~TNNFSANet() override = default;

  private:
    // In TNN: x*scale + bias
    static constexpr const float pad = 0.3f;
    std::vector<float> scale_vals = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    std::vector<float> bias_vals = {-1.f, -1.f, -1.f};

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::EulerAngles &euler_angles);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_FSANET_H
