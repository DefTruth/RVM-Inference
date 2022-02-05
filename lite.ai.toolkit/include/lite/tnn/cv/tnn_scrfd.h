//
// Created by DefTruth on 2021/12/30.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_SCRFD_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_SCRFD_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNSCRFD : public BasicTNNHandler
  {
  public:
    explicit TNNSCRFD(const std::string &_proto_path,
                      const std::string &_model_path,
                      unsigned int _num_threads = 1); //
    ~TNNSCRFD() override = default;

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
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f}; // RGB
    std::vector<float> bias_vals = {-127.5f / 128.f, -127.5f / 128.f, -127.5f / 128.f};
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
    void transform(const cv::Mat &mat_rs) override; // without resize

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
                                       std::shared_ptr<tnn::Mat> &score_pred,
                                       std::shared_ptr<tnn::Mat> &bbox_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::BoxfWithLandmarks> &bbox_kps_collection);

    void generate_bboxes_kps_single_stride(const SCRFDScaleParams &scale_params,
                                           std::shared_ptr<tnn::Mat> &score_pred,
                                           std::shared_ptr<tnn::Mat> &bbox_pred,
                                           std::shared_ptr<tnn::Mat> &kps_pred,
                                           unsigned int stride,
                                           float score_threshold,
                                           float img_height,
                                           float img_width,
                                           std::vector<types::BoxfWithLandmarks> &bbox_kps_collection);

    void generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                             std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                             std::shared_ptr<tnn::Instance> &_instance,
                             float score_threshold, float img_height,
                             float img_width); // rescale & exclude

    void nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                        std::vector<types::BoxfWithLandmarks> &output,
                        float iou_threshold, unsigned int topk);

  public:
    void detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 400);


  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_SCRFD_H
























































