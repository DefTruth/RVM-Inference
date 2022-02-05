//
// Created by DefTruth on 2021/12/5.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_MG_MATTING_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_MG_MATTING_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNMGMatting
  {
  public:
    explicit TNNMGMatting(const std::string &_proto_path,
                          const std::string &_model_path,
                          unsigned int _num_threads = 1);

    ~TNNMGMatting();

  private:
    const char *log_id = nullptr;
    const char *proto_path = nullptr;
    const char *model_path = nullptr;
    // Note, tnn:: actually is TNN_NS::, I prefer the first one.
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<tnn::Instance> instance;
    std::shared_ptr<tnn::Mat> image_mat;
    std::shared_ptr<tnn::Mat> mask_mat;

  private:
    std::vector<float> scale_vals = {(1.f / 0.229f) * (1.f / 255.f),
                                     (1.f / 0.224f) * (1.f / 255.f),
                                     (1.f / 0.225f) * (1.f / 255.f)};
    std::vector<float> bias_vals = {-0.485f * 255.f * (1.f / 0.229f) * (1.f / 255.f),
                                    -0.456f * 255.f * (1.f / 0.224f) * (1.f / 255.f),
                                    -0.406f * 255.f * (1.f / 0.225f) * (1.f / 255.f)}; // RGB

  private:
    const unsigned int num_threads; // initialize at runtime.
    int dynamic_input_height = 1024; // init only, will change according to input mat.
    int dynamic_input_width = 1024; // init only, will change according to input mat.
    tnn::DataFormat input_data_format;  // e.g DATA_FORMAT_NHWC
    tnn::MatType input_mat_type; // e.g NCHW_FLOAT
    tnn::DeviceType input_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType output_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType network_device_type; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM
    tnn::DimsVector image_shape; // debug
    tnn::DimsVector mask_shape; // debug
    tnn::DimsVector alpha_os1_shape; // debug
    tnn::DimsVector alpha_os4_shape; // debug
    tnn::DimsVector alpha_os8_shape; // debug
    static constexpr const unsigned int align_val = 32;

    // un-copyable
  protected:
    TNNMGMatting(const TNNMGMatting &) = delete; //
    TNNMGMatting(TNNMGMatting &&) = delete; //
    TNNMGMatting &operator=(const TNNMGMatting &) = delete; //
    TNNMGMatting &operator=(TNNMGMatting &&) = delete; //

  private:
    void print_debug_string();

  private:
    void transform(const cv::Mat &mat, const cv::Mat &mask);

    void initialize_instance(); // init net & instance

    cv::Mat padding(const cv::Mat &unpad_mat);

    void update_guidance_mask(cv::Mat &mask, unsigned int guidance_threshold = 128);

    void update_dynamic_shape(unsigned int img_height, unsigned int img_width);

    void update_alpha_pred(cv::Mat &alpha_pred, const cv::Mat &weight, const cv::Mat &other_alpha_pred);

    cv::Mat get_unknown_tensor_from_pred(const cv::Mat &alpha_pred, unsigned int rand_width = 30);

    void remove_small_connected_area(cv::Mat &alpha_pred);

    void generate_matting(std::shared_ptr<tnn::Instance> &_instance,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false);

  public:
    /**
     * Image Matting Using MGMatting(https://github.com/yucornetto/MGMatting)
     * @param mat: cv::Mat BGR HWC, source image
     * @param mask: cv::Mat Gray, guidance mask.
     * @param guidance_threshold: int, guidance threshold..
     * @param content: types::MattingContent to catch the detected results.
     */
    void detect(const cv::Mat &mat, cv::Mat &mask, types::MattingContent &content,
                bool remove_noise = false, unsigned int guidance_threshold = 128);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_MG_MATTING_H
