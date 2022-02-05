//
// Created by DefTruth on 2021/11/20.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_RETINAFACE_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_RETINAFACE_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNRetinaFace : public BasicNCNNHandler
  {
  public:
    explicit NCNNRetinaFace(const std::string &_param_path,
                            const std::string &_bin_path,
                            unsigned int _num_threads = 1,
                            int _input_height = 640,
                            int _input_width = 640);

    ~NCNNRetinaFace() override = default;

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
    const int input_height; // 640/320
    const int input_width; // 640/320

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
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

    void generate_anchors(const int target_height,
                          const int target_width,
                          std::vector<RetinaAnchor> &anchors);


    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         ncnn::Extractor &extractor,
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

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_RETINAFACE_H
