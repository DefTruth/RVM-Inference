//
// Created by DefTruth on 2021/10/7.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CORE_NCNN_HANDLER_H
#define LITE_AI_TOOLKIT_NCNN_CORE_NCNN_HANDLER_H

#include "ncnn_config.h"

namespace ncnncore
{
  class LITE_EXPORTS BasicNCNNHandler
  {
  protected:
    ncnn::Net *net = nullptr;
    const char *log_id = nullptr;
    const char *param_path = nullptr;
    const char *bin_path = nullptr;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;
    int num_outputs = 1;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  protected:
    explicit BasicNCNNHandler(const std::string &_param_path,
                              const std::string &_bin_path,
                              unsigned int _num_threads = 1);

    virtual ~BasicNCNNHandler();

    // un-copyable
  protected:
    BasicNCNNHandler(const BasicNCNNHandler &) = delete; //
    BasicNCNNHandler(BasicNCNNHandler &&) = delete; //
    BasicNCNNHandler &operator=(const BasicNCNNHandler &) = delete; //
    BasicNCNNHandler &operator=(BasicNCNNHandler &&) = delete; //

  private:
    virtual void transform(const cv::Mat &mat, ncnn::Mat &in) = 0;

  private:
    void initialize_handler();

    void print_debug_string();

  public:
    static void print_shape(const ncnn::Mat &mat, const std::string name = "");

  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CORE_NCNN_HANDLER_H
