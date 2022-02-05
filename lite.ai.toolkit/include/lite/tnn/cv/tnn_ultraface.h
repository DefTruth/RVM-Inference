//
// Created by DefTruth on 2021/11/20.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_ULTRAFACE_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_ULTRAFACE_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNUltraFace : public BasicTNNHandler
  {
  public:
    explicit TNNUltraFace(const std::string &_proto_path,
                          const std::string &_model_path,
                          unsigned int _num_threads = 1); //
    ~TNNUltraFace() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
    std::vector<float> bias_vals = {
        -127.0f * (1.0f / 128.0f),
        -127.0f * (1.0f / 128.0f),
        -127.0f * (1.0f / 128.0f)
    }; // RGB
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    void transform(const cv::Mat &mat) override; //

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::shared_ptr<tnn::Instance> &_instance,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.7f, float iou_threshold = 0.3f,
                unsigned int topk = 300, unsigned int nms_type = 0);

  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_ULTRAFACE_H
