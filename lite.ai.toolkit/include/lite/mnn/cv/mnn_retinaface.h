//
// Created by DefTruth on 2021/11/20.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_RETINAFACE_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_RETINAFACE_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNRetinaFace : public BasicMNNHandler
  {
  public:
    explicit MNNRetinaFace(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNRetinaFace() override = default;

  private:
    // nested classes
    struct RetinaAnchor
    {
      float cx;
      float cy;
      float s_kx;
      float s_ky;
    };

  private:
    const float mean_vals[3] = {104.f, 117.f, 123.f}; // bgr order
    const float norm_vals[3] = {1.f, 1.f, 1.f};
    const float variance[2] = {0.1f, 0.2f};
    std::vector<int> steps = {8, 16, 32};
    std::vector<std::vector<int>> min_sizes = {
        {16,  32},
        {64,  128},
        {256, 512}
    };

    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; //

    void generate_anchors(const int target_height,
                          const int target_width,
                          std::vector<RetinaAnchor> &anchors);

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         const std::map<std::string, MNN::Tensor *> &output_tensors,
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

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_RETINAFACE_H
