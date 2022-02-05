//
// Created by DefTruth on 2021/12/5.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MG_MATTING_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MG_MATTING_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMGMatting
  {
  public:
    explicit MNNMGMatting(const std::string &_mnn_path, unsigned int _num_threads = 8); //
    ~MNNMGMatting();

  private:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::ScheduleConfig schedule_config;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at runtime
    const char *log_id = nullptr;
    const char *mnn_path = nullptr;
    MNN::Tensor *image_tensor = nullptr;
    MNN::Tensor *mask_tensor = nullptr;

  private:
    const float norm_vals[3] = {(1.f / 0.229f) * (1.f / 255.f),
                                (1.f / 0.224f) * (1.f / 255.f),
                                (1.f / 0.225f) * (1.f / 255.f)};
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; // RGB

  private:
    const unsigned int num_threads; // initialize at runtime.
    int dynamic_input_height = 512; // init only, will change according to input mat.
    int dynamic_input_width = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_image_size = 1 * 3 * 512 * 512; // init only, will change according to input mat.
    unsigned int dynamic_input_mask_size = 1 * 1 * 512 * 512; // init only, will change according to input mat.
    int dimension_type; // hint only
    static constexpr const unsigned int align_val = 32;

    // un-copyable
  protected:
    MNNMGMatting(const MNNMGMatting &) = delete; //
    MNNMGMatting(MNNMGMatting &&) = delete; //
    MNNMGMatting &operator=(const MNNMGMatting &) = delete; //
    MNNMGMatting &operator=(MNNMGMatting &&) = delete; //

  private:
    void print_debug_string();

  private:
    void transform(const cv::Mat &mat, const cv::Mat &mask);

    void initialize_pretreat(); //

    void initialize_interpreter();

    cv::Mat padding(const cv::Mat &unpad_mat);

    void update_guidance_mask(cv::Mat &mask, unsigned int guidance_threshold = 128);

    void update_dynamic_shape(unsigned int img_height, unsigned int img_width);

    void update_alpha_pred(cv::Mat &alpha_pred, const cv::Mat &weight, const cv::Mat &other_alpha_pred);

    cv::Mat get_unknown_tensor_from_pred(const cv::Mat &alpha_pred, unsigned int rand_width = 30);

    void remove_small_connected_area(cv::Mat &alpha_pred);

    void generate_matting(const std::map<std::string, MNN::Tensor *> &output_tensors,
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


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MG_MATTING_H
