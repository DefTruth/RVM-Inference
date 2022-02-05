//
// Created by DefTruth on 2021/11/14.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_TENCENT_CURRICULAR_FACE_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_TENCENT_CURRICULAR_FACE_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNTencentCurricularFace : public BasicTNNHandler
  {
  public:
    explicit TNNTencentCurricularFace(const std::string &_proto_path,
                                      const std::string &_model_path,
                                      unsigned int _num_threads = 1); //
    ~TNNTencentCurricularFace() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    std::vector<float> bias_vals = {-1.f, -1.f, -1.f}; // RGB

  private:
    void transform(const cv::Mat &mat) override; //

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content);

  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_TENCENT_CURRICULAR_FACE_H
