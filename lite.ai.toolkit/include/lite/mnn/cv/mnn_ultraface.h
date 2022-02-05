//
// Created by DefTruth on 2021/11/20.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_ULTRAFACE_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_ULTRAFACE_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNUltraFace : public BasicMNNHandler
  {
  public:
    explicit MNNUltraFace(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNUltraFace() override = default;

  private:
    const float mean_vals[3] = {127.0f, 127.0f, 127.0f};
    const float norm_vals[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    void initialize_pretreat(); //

    void transform(const cv::Mat &mat) override; //

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

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_ULTRAFACE_H
