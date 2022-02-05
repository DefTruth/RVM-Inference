//
// Created by DefTruth on 2021/10/7.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CORE_NCNN_CORE_H
#define LITE_AI_TOOLKIT_NCNN_CORE_NCNN_CORE_H

#include "ncnn_config.h"
#include "ncnn_handler.h"
#include "ncnn_types.h"
#include "ncnn_custom.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNNanoDet;                                // [0] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS NCNNNanoDetEfficientNetLite;                // [1] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS NCNNNanoDetDepreciated;                     // [2] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS NCNNNanoDetEfficientNetLiteDepreciated;     // [3] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS NCNNRobustVideoMatting;                     // [4] * reference: https://github.com/PeterL1n/RobustVideoMatting
  class LITE_EXPORTS NCNNYoloX;                                  // [5] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS NCNNYOLOP;                                  // [6] * reference: https://github.com/hustvl/YOLOP
  class LITE_EXPORTS NCNNYoloV5;                                 // [7] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS NCNNYoloX_V_0_1_1;                          // [8] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS NCNNYoloR;                                  // [9] * reference: https://github.com/WongKinYiu/yolor
  class LITE_EXPORTS NCNNYoloRssss;                              // [10] * reference: https://github.com/WongKinYiu/yolor
  class LITE_EXPORTS NCNNYoloV5_V_6_0;                           // [11] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS NCNNYoloV5_V_6_0_P6;                        // [12] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS NCNNGlintArcFace;                           // [13] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS NCNNGlintCosFace;                           // [14] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS NCNNGlintPartialFC;                         // [15] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc
  class LITE_EXPORTS NCNNFaceNet;                                // [16] * reference: https://github.com/timesler/facenet-pytorch
  class LITE_EXPORTS NCNNFocalArcFace;                           // [17] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS NCNNFocalAsiaArcFace;                       // [18] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS NCNNTencentCurricularFace;                  // [19] * reference: https://github.com/Tencent/TFace/tree/master/tasks/distfc
  class LITE_EXPORTS NCNNTencentCifpFace;                        // [20] * reference: https://github.com/Tencent/TFace/tree/master/tasks/cifp
  class LITE_EXPORTS NCNNCenterLossFace;                         // [21] * reference: https://github.com/louis-she/center-loss.pytorch
  class LITE_EXPORTS NCNNSphereFace;                             // [22] * reference: https://github.com/clcarwin/sphereface_pytorch
  class LITE_EXPORTS NCNNMobileFaceNet;                          // [23] * reference: https://github.com/Xiaoccer/MobileFaceNet_Pytorch
  class LITE_EXPORTS NCNNCavaGhostArcFace;                       // [24] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS NCNNCavaCombinedFace;                       // [25] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS NCNNMobileSEFocalFace;                      // [26] * reference: https://github.com/grib0ed0v/face_recognition.pytorch
  class LITE_EXPORTS NCNNUltraFace;                              // [27] * reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
  class LITE_EXPORTS NCNNRetinaFace;                             // [28] * reference: https://github.com/biubug6/Pytorch_Retinaface
  class LITE_EXPORTS NCNNFaceBoxes;                              // [29] * reference: https://github.com/zisianw/FaceBoxes.PyTorch
  class LITE_EXPORTS NCNNPFLD;                                   // [30] * reference: https://github.com/Hsintao/pfld_106_face_landmarks
  class LITE_EXPORTS NCNNPFLD98;                                 // [31] * reference: https://github.com/polarisZhao/PFLD-pytorch
  class LITE_EXPORTS NCNNMobileNetV268;                          // [32] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS NCNNMobileNetV2SE68;                        // [33] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS NCNNPFLD68;                                 // [34] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS NCNNFaceLandmark1000;                       // [35] * reference: https://github.com/Single430/FaceLandmark1000
  class LITE_EXPORTS NCNNAgeGoogleNet;                           // [36] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS NCNNGenderGoogleNet;                        // [37] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS NCNNEmotionFerPlus;                         // [38] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus
  class LITE_EXPORTS NCNNEfficientEmotion7;                      // [39] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS NCNNEfficientEmotion8;                      // [40] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS NCNNMobileEmotion7;                         // [41] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS NCNNEfficientNetLite4;                      // [42] * reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class LITE_EXPORTS NCNNShuffleNetV2;                           // [43] * reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class LITE_EXPORTS NCNNDenseNet;                               // [44] * reference: https://pytorch.org/hub/pytorch_vision_densenet/
  class LITE_EXPORTS NCNNGhostNet;                               // [45] * reference：https://pytorch.org/hub/pytorch_vision_ghostnet/
  class LITE_EXPORTS NCNNHdrDNet;                                // [46] * reference: https://pytorch.org/hub/pytorch_vision_hardnet/
  class LITE_EXPORTS NCNNIBNNet;                                 // [47] * reference: https://pytorch.org/hub/pytorch_vision_ibnnet/
  class LITE_EXPORTS NCNNMobileNetV2;                            // [48] * reference: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
  class LITE_EXPORTS NCNNResNet;                                 // [49] * reference: https://pytorch.org/hub/pytorch_vision_resnet/
  class LITE_EXPORTS NCNNResNeXt;                                // [50] * reference: https://pytorch.org/hub/pytorch_vision_resnext/
  class LITE_EXPORTS NCNNFastStyleTransfer;                      // [51] * reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class LITE_EXPORTS NCNNColorizer;                              // [52] * reference: https://github.com/richzhang/colorization
  class LITE_EXPORTS NCNNSubPixelCNN;                            // [53] * reference: https://github.com/niazwazir/SUB_PIXEL_CNN
  class LITE_EXPORTS NCNNDeepLabV3ResNet101;                     // [54] * reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  class LITE_EXPORTS NCNNFCNResNet101;                           // [55] * reference: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
  class LITE_EXPORTS NCNNNanoDetPlus;                            // [56] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS NCNNSCRFD;                                  // [57] * reference: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
  class LITE_EXPORTS NCNNYOLO5Face;                              // [58] * reference: https://github.com/deepcam-cn/yolov5-face
}

namespace ncnncv
{
  using ncnncore::BasicNCNNHandler;
}

namespace ncnnnlp
{
  using ncnncore::BasicNCNNHandler;
}

namespace ncnnasr
{
  using ncnncore::BasicNCNNHandler;
}

#endif //LITE_AI_TOOLKIT_NCNN_CORE_NCNN_CORE_H
