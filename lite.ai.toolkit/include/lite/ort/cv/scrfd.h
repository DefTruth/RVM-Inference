//
// Created by DefTruth on 2021/12/30.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_SCRFD_H
#define LITE_AI_TOOLKIT_ORT_CV_SCRFD_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  // https://github.com/deepinsight/insightface/blob/master/detection/scrfd/
  // mmdet/core/anchor/anchor_generator.py
  class LITE_EXPORTS SCRFD : public BasicOrtHandler
  {
  public:
    explicit SCRFD(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~SCRFD() override = default;

  private:
    // nested classes
    typedef struct
    {
      float cx;
      float cy;
      float stride;
    } SCRFDPoint;
    typedef struct
    {
      float ratio;
      int dw;
      int dh;
      bool flag;
    } SCRFDScaleParams;

  private:
    // blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float scale_vals[3] = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
    unsigned int fmc = 3; // feature map count
    bool use_kps = false;
    unsigned int num_anchors = 2;
    std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]
    // if num_anchors>1, then stack points in col major -> (height*num_anchor*width,2)
    // anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
    std::unordered_map<int, std::vector<SCRFDPoint>> center_points;
    bool center_points_is_update = false;
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override; // without resize

    // initial steps and num_anchors
    // https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py
    void initial_context();

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        SCRFDScaleParams &scale_params);

    // generate once.
    void generate_points(const int target_height, const int target_width);

    void generate_bboxes_single_stride(const SCRFDScaleParams &scale_params,
                                       Ort::Value &score_pred,
                                       Ort::Value &bbox_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::BoxfWithLandmarks> &bbox_kps_collection);

    void generate_bboxes_kps_single_stride(const SCRFDScaleParams &scale_params,
                                           Ort::Value &score_pred,
                                           Ort::Value &bbox_pred,
                                           Ort::Value &kps_pred,
                                           unsigned int stride,
                                           float score_threshold,
                                           float img_height,
                                           float img_width,
                                           std::vector<types::BoxfWithLandmarks> &bbox_kps_collection);

    void generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                             std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                             std::vector<Ort::Value> &output_tensors,
                             float score_threshold, float img_height,
                             float img_width); // rescale & exclude

    void nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                        std::vector<types::BoxfWithLandmarks> &output,
                        float iou_threshold, unsigned int topk);

  public:
    void detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                float score_threshold = 0.3f, float iou_threshold = 0.45f,
                unsigned int topk = 400);

  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_SCRFD_H
