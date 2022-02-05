//
// Created by DefTruth on 2021/11/21.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_LANDMARKS_1000_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_LANDMARKS_1000_H


#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFaceLandmark1000 : public BasicMNNHandler
  {
  public:
    explicit MNNFaceLandmark1000(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNFaceLandmark1000() override = default;

  private:
    const float mean_vals[1] = {0.0f};
    const float norm_vals[1] = {1.0f};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);
  };
}
#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_LANDMARKS_1000_H
