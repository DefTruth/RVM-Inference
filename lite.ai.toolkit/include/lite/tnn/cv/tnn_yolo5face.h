//
// Created by DefTruth on 2022/1/16.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_YOLO5FACE_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_YOLO5FACE_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNYOLO5Face : public BasicTNNHandler
  {
  public:
    explicit TNNYOLO5Face(const std::string &_proto_path,
                          const std::string &_model_path,
                          unsigned int _num_threads = 1);

    ~TNNYOLO5Face() override = default;

  private:
    // nested classes
    typedef struct
    {
      float ratio;
      int dw;
      int dh;
      bool flag;
    } YOLO5FaceScaleParams;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    std::vector<float> bias_vals = {0.f, 0.f, 0.f}; // RGB
    static constexpr const unsigned int max_nms = 30000;

  private:
    void transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YOLO5FaceScaleParams &scale_params);

    void generate_bboxes_kps(const YOLO5FaceScaleParams &scale_params,
                             std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                             std::shared_ptr<tnn::Instance> &_instance,
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

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_YOLO5FACE_H
