//
// Created by DefTruth on 2021/9/20.
//

#include "lite/lite.h"

// Video Matting Interface
static void test_video()
{
  std::string onnx_path = "../hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
  std::string video_path = "../resources/interview.mp4";
  std::string output_path = "../logs/interview_onnx.mp4";

  auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
  std::vector<lite::cv::types::MattingContent> contents;

  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents, false, 0.25f);

  delete rvm;
}


// Image Matting Interface
static void test_image()
{
  std::string onnx_path = "../hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
  std::string img_path = "../resources/test.jpg";
  std::string save_fgr_path = "../logs/test_rvm_fgr.jpg";
  std::string save_pha_path = "../logs/test_rvm_pha.jpg";
  std::string save_merge_path = "../logs/test_rvm_merge.jpg";

  auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
  lite::cv::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(img_path);

  // 1. image matting.
  rvm->detect(img_bgr, content, 0.25f);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
  }


  delete rvm;
}


static void test_rvm()
{
  test_video();
  test_image();
}

int main(__unused int argc, __unused char *argv[])
{
  test_rvm();
  return 0;
}
