//
// Created by DefTruth on 2021/10/18.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOP_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOP_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNYOLOP
  {
  private:
    ncnn::Net *net = nullptr;
    const char *log_id = nullptr;
    const char *param_path = nullptr;
    const char *bin_path = nullptr;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;

  public:
    explicit NCNNYOLOP(const std::string &_param_path,
                       const std::string &_bin_path,
                       unsigned int _num_threads = 1,
                       int _input_height = 640,
                       int _input_width = 640); //
    ~NCNNYOLOP();

  private:
    // nested classes
    typedef struct
    {
      int grid0;
      int grid1;
      int stride;
      float width;
      float height;
    } YOLOPAnchor;

    typedef struct
    {
      float r;
      int dw;
      int dh;
      int new_unpad_w;
      int new_unpad_h;
      bool flag;
    } YOLOPScaleParams;

  private:
    const unsigned int num_threads; // initialize at runtime.
    // target image size after resize
    const int input_height; // 640/320/1280
    const int input_width; // 640/320/1280

    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f}; // RGB
    const float norm_vals[3] = {1.f / (255.f * 0.229f), 1.f / (255.f * 0.224f), 1.f / (255.f * 0.225f)};

    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;

    std::vector<unsigned int> strides = {8, 16, 32};
    std::unordered_map<unsigned int, std::vector<YOLOPAnchor>> center_anchors;
    bool center_anchors_is_update = false;

  protected:
    NCNNYOLOP(const NCNNYOLOP &) = delete; //
    NCNNYOLOP(NCNNYOLOP &&) = delete; //
    NCNNYOLOP &operator=(const NCNNYOLOP &) = delete; //
    NCNNYOLOP &operator=(NCNNYOLOP &&) = delete; //

  private:
    void print_debug_string();

    void transform(const cv::Mat &mat_rs, ncnn::Mat &in);

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YOLOPScaleParams &scale_params);

    // only generate once
    void generate_anchors(unsigned int target_height, unsigned int target_width);

    void generate_bboxes_single_stride(const YOLOPScaleParams &scale_params,
                                       ncnn::Mat &det_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::Boxf> &bbox_collection);

    void generate_bboxes_da_ll(const YOLOPScaleParams &scale_params,
                               ncnn::Extractor &extractor,
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

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLOP_H
