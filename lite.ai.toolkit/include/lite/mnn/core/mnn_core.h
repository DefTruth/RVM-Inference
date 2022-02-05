//
// Created by DefTruth on 2021/10/6.
//

#ifndef LITE_AI_TOOLKIT_MNN_CORE_MNN_CORE_H
#define LITE_AI_TOOLKIT_MNN_CORE_MNN_CORE_H

#include "mnn_config.h"
#include "mnn_handler.h"
#include "mnn_types.h"

namespace mnncv
{
  class LITE_EXPORTS MNNNanoDet;                     // [0] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS MNNNanoDetEfficientNetLite;     // [1] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS MNNRobustVideoMatting;          // [2] * reference: https://github.com/PeterL1n/RobustVideoMatting
  class LITE_EXPORTS MNNYoloX;                       // [3] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS MNNYOLOP;                       // [4] * reference: https://github.com/hustvl/YOLOP
  class LITE_EXPORTS MNNYoloV5;                      // [5] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS MNNYoloX_V_0_1_1;               // [6] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS MNNYoloR;                       // [7] * reference: https://github.com/WongKinYiu/yolor
  class LITE_EXPORTS MNNYoloV5_V_6_0;                // [8] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS MNNGlintArcFace;                // [9] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS MNNGlintCosFace;                // [10] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS MNNGlintPartialFC;              // [11] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc
  class LITE_EXPORTS MNNFaceNet;                     // [12] * reference: https://github.com/timesler/facenet-pytorch
  class LITE_EXPORTS MNNFocalArcFace;                // [13] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS MNNFocalAsiaArcFace;            // [14] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS MNNTencentCurricularFace;       // [15] * reference: https://github.com/Tencent/TFace/tree/master/tasks/distfc
  class LITE_EXPORTS MNNTencentCifpFace;             // [16] * reference: https://github.com/Tencent/TFace/tree/master/tasks/cifp
  class LITE_EXPORTS MNNCenterLossFace;              // [17] * reference: https://github.com/louis-she/center-loss.pytorch
  class LITE_EXPORTS MNNSphereFace;                  // [18] * reference: https://github.com/clcarwin/sphereface_pytorch
  class LITE_EXPORTS MNNMobileFaceNet;               // [19] * reference: https://github.com/Xiaoccer/MobileFaceNet_Pytorch
  class LITE_EXPORTS MNNCavaGhostArcFace;            // [20] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS MNNCavaCombinedFace;            // [21] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS MNNMobileSEFocalFace;           // [22] * reference: https://github.com/grib0ed0v/face_recognition.pytorch
  class LITE_EXPORTS MNNUltraFace;                   // [23] * reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
  class LITE_EXPORTS MNNRetinaFace;                  // [24] * reference: https://github.com/biubug6/Pytorch_Retinaface
  class LITE_EXPORTS MNNFaceBoxes;                   // [25] * reference: https://github.com/zisianw/FaceBoxes.PyTorch
  class LITE_EXPORTS MNNPFLD;                        // [26] * reference: https://github.com/Hsintao/pfld_106_face_landmarks
  class LITE_EXPORTS MNNPFLD98;                      // [27] * reference: https://github.com/polarisZhao/PFLD-pytorch
  class LITE_EXPORTS MNNMobileNetV268;               // [28] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS MNNMobileNetV2SE68;             // [29] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS MNNPFLD68;                      // [30] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS MNNFaceLandmark1000;            // [31] * reference: https://github.com/Single430/FaceLandmark1000
  class LITE_EXPORTS MNNFSANet;                      // [32] * reference: https://github.com/omasaht/headpose-fsanet-pytorch
  class LITE_EXPORTS MNNAgeGoogleNet;                // [33] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS MNNGenderGoogleNet;             // [34] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS MNNEmotionFerPlus;              // [35] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus
  class LITE_EXPORTS MNNSSRNet;                      // [36] * reference: https://github.com/oukohou/SSR_Net_Pytorch
  class LITE_EXPORTS MNNEfficientEmotion7;           // [37] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS MNNEfficientEmotion8;           // [38] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS MNNMobileEmotion7;              // [39] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS MNNReXNetEmotion7;              // [40] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS MNNEfficientNetLite4;           // [41] * reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class LITE_EXPORTS MNNShuffleNetV2;                // [42] * reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class LITE_EXPORTS MNNDenseNet;                    // [43] * reference: https://pytorch.org/hub/pytorch_vision_densenet/
  class LITE_EXPORTS MNNGhostNet;                    // [44] * reference：https://pytorch.org/hub/pytorch_vision_ghostnet/
  class LITE_EXPORTS MNNHdrDNet;                     // [45] * reference: https://pytorch.org/hub/pytorch_vision_hardnet/
  class LITE_EXPORTS MNNIBNNet;                      // [46] * reference: https://pytorch.org/hub/pytorch_vision_ibnnet/
  class LITE_EXPORTS MNNMobileNetV2;                 // [47] * reference: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
  class LITE_EXPORTS MNNResNet;                      // [48] * reference: https://pytorch.org/hub/pytorch_vision_resnet/
  class LITE_EXPORTS MNNResNeXt;                     // [49] * reference: https://pytorch.org/hub/pytorch_vision_resnext/
  class LITE_EXPORTS MNNFastStyleTransfer;           // [50] * reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class LITE_EXPORTS MNNColorizer;                   // [51] * reference: https://github.com/richzhang/colorization
  class LITE_EXPORTS MNNSubPixelCNN;                 // [52] * reference: https://github.com/niazwazir/SUB_PIXEL_CNN
  class LITE_EXPORTS MNNDeepLabV3ResNet101;          // [53] * reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  class LITE_EXPORTS MNNFCNResNet101;                // [54] * reference: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
  class LITE_EXPORTS MNNMGMatting;                   // [55] * reference: https://github.com/yucornetto/MGMatting
  class LITE_EXPORTS MNNNanoDetPlus;                 // [56] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS MNNSCRFD;                       // [57] * reference: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
  class LITE_EXPORTS MNNYOLO5Face;                   // [58] * reference: https://github.com/deepcam-cn/yolov5-face

}

namespace mnncv
{
  using mnncore::BasicMNNHandler;
}

namespace mnnnlp
{
  using mnncore::BasicMNNHandler;
}

namespace mnnasr
{
  using mnncore::BasicMNNHandler;
}

#endif //LITE_AI_TOOLKIT_MNN_CORE_MNN_CORE_H
