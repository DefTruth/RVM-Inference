//
// Created by DefTruth on 2021/10/18.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_YOLOP_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_YOLOP_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  typedef struct
  {
    float r;
    int dw;
    int dh;
    int new_unpad_w;
    int new_unpad_h;
    bool flag;
  } YOLOPScaleParams;

  class LITE_EXPORTS TNNYOLOP : public BasicTNNHandler
  {
  public:
    explicit TNNYOLOP(const std::string &_proto_path,
                      const std::string &_model_path,
                      unsigned int _num_threads = 1); //
    ~TNNYOLOP() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {0.0171247f, 0.0175070f, 0.0174291f}; // RGB
    std::vector<float> bias_vals = {-123.675f * 0.0171247f, -116.28f * 0.0175070f, -103.53f * 0.0174291f};

    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    void transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YOLOPScaleParams &scale_params);

    void generate_bboxes_da_ll(const YOLOPScaleParams &scale_params,
                               std::shared_ptr<tnn::Instance> &_instance,
                               std::vector<types::Boxf> &bbox_collection,
                               types::SegmentContent &da_seg_content,
                               types::SegmentContent &ll_seg_content,
                               float score_threshold, float img_height,
                               float img_width); // det,da_seg,ll_seg

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat,
                std::vector<types::Boxf> &detected_boxes,
                types::SegmentContent &da_seg_content,
                types::SegmentContent &ll_seg_content,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

  };

}


#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_YOLOP_H
