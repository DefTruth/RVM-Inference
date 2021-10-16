//
// Created by DefTruth on 2021/9/14.
//

#ifndef LITE_AI_ORT_CV_YOLOP_H
#define LITE_AI_ORT_CV_YOLOP_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
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

  class LITE_EXPORTS YOLOP : public BasicOrtHandler
  {
  public:
    explicit YOLOP(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~YOLOP() override = default;

  private:
    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float scale_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};

    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YOLOPScaleParams &scale_params);

    void generate_bboxes_da_ll(const YOLOPScaleParams &scale_params,
                               std::vector<Ort::Value> &output_tensors,
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


#endif //LITE_AI_ORT_CV_YOLOP_H
