//
// Created by DefTruth on 2022/1/16.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLO5FACE_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLO5FACE_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNYOLO5Face
  {
  private:
    ncnn::Net *net = nullptr;
    const char *log_id = nullptr;
    const char *param_path = nullptr;
    const char *bin_path = nullptr;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;

  private:
    // nested classes
    typedef struct
    {
      int grid0;
      int grid1;
      int stride;
      float width;
      float height;
    } YOLO5FaceAnchor;

    typedef struct
    {
      float ratio;
      int dw;
      int dh;
      bool flag;
    } YOLO5FaceScaleParams;

  public:
    explicit NCNNYOLO5Face(const std::string &_param_path,
                           const std::string &_bin_path,
                           unsigned int _num_threads = 1,
                           int _input_height = 640,
                           int _input_width = 640); //
    ~NCNNYOLO5Face();

  private:
    const unsigned int num_threads; // initialize at runtime.
    // target image size after resize
    const int input_height; // 640
    const int input_width; // 640

    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.0 / 255.f, 1.0 / 255.f, 1.0 / 255.f};
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;

    std::vector<unsigned int> strides = {8, 16, 32};
    std::unordered_map<unsigned int, std::vector<YOLO5FaceAnchor>> center_anchors;
    bool center_anchors_is_update = false;

  protected:
    NCNNYOLO5Face(const NCNNYOLO5Face &) = delete; //
    NCNNYOLO5Face(NCNNYOLO5Face &&) = delete; //
    NCNNYOLO5Face &operator=(const NCNNYOLO5Face &) = delete; //
    NCNNYOLO5Face &operator=(NCNNYOLO5Face &&) = delete; //

  private:
    void print_debug_string();

    void transform(const cv::Mat &mat_rs, ncnn::Mat &in);

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YOLO5FaceScaleParams &scale_params);

    // only generate once
    void generate_anchors(unsigned int target_height, unsigned int target_width);

    void generate_bboxes_kps_single_stride(const YOLO5FaceScaleParams &scale_params,
                                           ncnn::Mat &det_pred,
                                           unsigned int stride,
                                           float score_threshold,
                                           float img_height,
                                           float img_width,
                                           std::vector<types::BoxfWithLandmarks> &bbox_kps_collection);

    void generate_bboxes_kps(const YOLO5FaceScaleParams &scale_params,
                             std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                             ncnn::Extractor &extractor,
                             float score_threshold, float img_height,
                             float img_width);

    void nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                        std::vector<types::BoxfWithLandmarks> &output,
                        float iou_threshold, unsigned int topk);

  public:
    void detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 400);

  };

}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_YOLO5FACE_H
