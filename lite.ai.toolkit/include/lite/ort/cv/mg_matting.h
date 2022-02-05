//
// Created by DefTruth on 2021/12/5.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_MG_MATTING_H
#define LITE_AI_TOOLKIT_ORT_CV_MG_MATTING_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS MGMatting
  {
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    // CPU MemoryInfo
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    // hardcode input node names
    const unsigned int num_inputs = 2;
    std::vector<const char *> input_node_names = {"image", "mask"};
    // hardcode output node names
    const unsigned int num_outputs = 3;
    std::vector<const char *> output_node_names = {
        "alpha_os1", "alpha_os4", "alpha_os8"
    };
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;

    // input values handler
    unsigned int dynamic_input_height = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_width = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_image_size = 1 * 3 * 512 * 512; // init only, will change according to input mat.
    unsigned int dynamic_input_mask_size = 1 * 1 * 512 * 512; // init only, will change according to input mat.
    std::vector<int64_t> dynamic_input_image_dims = {1, 3, 512, 512}; // init only, will change according to input mat.
    std::vector<int64_t> dynamic_input_mask_dims = {1, 1, 512, 512}; // init only, will change according to input mat.
    std::vector<float> dynamic_image_values_handler;
    std::vector<float> dynamic_mask_values_handler;

  private:
    const unsigned int num_threads; // initialize at runtime.
    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    const float scale_vals[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f}; // RGB
    static constexpr const unsigned int align_val = 32;

  public:
    explicit MGMatting(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~MGMatting();

  protected:
    MGMatting(const MGMatting &) = delete; //
    MGMatting(MGMatting &&) = delete; //
    MGMatting &operator=(const MGMatting &) = delete; //
    MGMatting &operator=(MGMatting &&) = delete; //

  private:
    // return normalized image,mask Tensors
    std::vector<Ort::Value> transform(const cv::Mat &mat, const cv::Mat &mask);

    void print_debug_string();

    cv::Mat padding(const cv::Mat &unpad_mat);

    void update_guidance_mask(cv::Mat &mask, unsigned int guidance_threshold = 128);

    void update_dynamic_shape(unsigned int img_height, unsigned int img_width);

    void update_alpha_pred(cv::Mat &alpha_pred, const cv::Mat &weight, const cv::Mat &other_alpha_pred);

    cv::Mat get_unknown_tensor_from_pred(const cv::Mat &alpha_pred, unsigned int rand_width = 30);

    void remove_small_connected_area(cv::Mat &alpha_pred);

    void generate_matting(std::vector<Ort::Value> &output_tensors,
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


#endif //LITE_AI_TOOLKIT_ORT_CV_MG_MATTING_H
