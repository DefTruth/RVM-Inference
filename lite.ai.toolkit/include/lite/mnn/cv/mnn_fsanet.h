//
// Created by DefTruth on 2021/11/25.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FSANET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FSANET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFSANet : public BasicMNNHandler
  {
  public:
    explicit MNNFSANet(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNFSANet() override = default;

  private:
    static constexpr const float pad = 0.3f;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; // padding & resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::EulerAngles &euler_angles);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FSANET_H
