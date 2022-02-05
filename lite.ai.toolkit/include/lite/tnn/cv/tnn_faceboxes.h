//
// Created by DefTruth on 2021/11/20.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_FACEBOXES_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_FACEBOXES_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNFaceBoxes : public BasicTNNHandler
  {
  public:
    explicit TNNFaceBoxes(const std::string &_proto_path,
                          const std::string &_model_path,
                          unsigned int _num_threads = 1); //
    ~TNNFaceBoxes() override = default;

  private:
    // nested classes
    struct FaceBoxesAnchor
    {
      float cx;
      float cy;
      float s_kx;
      float s_ky;
    };

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f, 1.f, 1.f};
    std::vector<float> bias_vals = {
        -104.f * 1.0f,
        -117.f * 1.0f,
        -123.f * 1.0f
    }; // bgr order
    const float variance[2] = {0.1f, 0.2f};
    std::vector<int> steps = {32, 64, 128};
    std::vector<std::vector<int>> min_sizes = {
        {32, 64, 128},
        {256},
        {512}
    };

    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    void transform(const cv::Mat &mat) override; //

    void generate_anchors(const int target_height,
                          const int target_width,
                          std::vector<FaceBoxesAnchor> &anchors);

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

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_FACEBOXES_H
