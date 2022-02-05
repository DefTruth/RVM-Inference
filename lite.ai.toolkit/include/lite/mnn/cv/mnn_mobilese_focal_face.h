//
// Created by DefTruth on 2021/11/14.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILESE_FOCAL_FACE_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILESE_FOCAL_FACE_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMobileSEFocalFace : public BasicMNNHandler
  {
  public:
    explicit MNNMobileSEFocalFace(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNMobileSEFocalFace() override = default;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.f / 255.0f, 1.f / 255.0f, 1.f / 255.0f};

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content);
  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILESE_FOCAL_FACE_H
