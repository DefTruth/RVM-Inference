//
// Created by DefTruth on 2021/10/17.
//

#ifndef LITE_AI_TOOLKIT_TNN_CORE_TNN_CORE_H
#define LITE_AI_TOOLKIT_TNN_CORE_TNN_CORE_H

#include "tnn_config.h"
#include "tnn_handler.h"
#include "tnn_types.h"

namespace tnncv
{
  class LITE_EXPORTS TNNNanoDet;                     // [0] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS TNNNanoDetEfficientNetLite;     // [1] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS TNNRobustVideoMatting;          // [2] * reference: https://github.com/PeterL1n/RobustVideoMatting
  class LITE_EXPORTS TNNYoloX;                       // [3] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS TNNYOLOP;                       // [4] * reference: https://github.com/hustvl/YOLOP
  class LITE_EXPORTS TNNYoloV5;                      // [5] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS TNNYoloX_V_0_1_1;               // [6] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS TNNYoloR;                       // [7] * reference: https://github.com/WongKinYiu/yolor
  class LITE_EXPORTS TNNYoloV5_V_6_0;                // [8] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS TNNGlintArcFace;                // [9] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS TNNGlintCosFace;                // [10] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS TNNGlintPartialFC;              // [11] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc
  class LITE_EXPORTS TNNFaceNet;                     // [12] * reference: https://github.com/timesler/facenet-pytorch
  class LITE_EXPORTS TNNFocalArcFace;                // [13] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS TNNFocalAsiaArcFace;            // [14] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS TNNTencentCurricularFace;       // [15] * reference: https://github.com/Tencent/TFace/tree/master/tasks/distfc
  class LITE_EXPORTS TNNTencentCifpFace;             // [16] * reference: https://github.com/Tencent/TFace/tree/master/tasks/cifp
  class LITE_EXPORTS TNNCenterLossFace;              // [17] * reference: https://github.com/louis-she/center-loss.pytorch
  class LITE_EXPORTS TNNSphereFace;                  // [18] * reference: https://github.com/clcarwin/sphereface_pytorch
  class LITE_EXPORTS TNNMobileFaceNet;               // [19] * reference: https://github.com/Xiaoccer/MobileFaceNet_Pytorch
  class LITE_EXPORTS TNNCavaGhostArcFace;            // [20] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS TNNCavaCombinedFace;            // [21] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS TNNMobileSEFocalFace;           // [22] * reference: https://github.com/grib0ed0v/face_recognition.pytorch
  class LITE_EXPORTS TNNUltraFace;                   // [23] * reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
  class LITE_EXPORTS TNNRetinaFace;                  // [24] * reference: https://github.com/biubug6/Pytorch_Retinaface
  class LITE_EXPORTS TNNFaceBoxes;                   // [25] * reference: https://github.com/zisianw/FaceBoxes.PyTorch
  class LITE_EXPORTS TNNPFLD;                        // [26] * reference: https://github.com/Hsintao/pfld_106_face_landmarks
  class LITE_EXPORTS TNNPFLD98;                      // [27] * reference: https://github.com/polarisZhao/PFLD-pytorch
  class LITE_EXPORTS TNNMobileNetV268;               // [28] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS TNNMobileNetV2SE68;             // [29] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS TNNPFLD68;                      // [30] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS TNNFaceLandmark1000;            // [31] * reference: https://github.com/Single430/FaceLandmark1000
  class LITE_EXPORTS TNNFSANet;                      // [32] * reference: https://github.com/omasaht/headpose-fsanet-pytorch
  class LITE_EXPORTS TNNAgeGoogleNet;                // [33] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS TNNGenderGoogleNet;             // [34] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS TNNEmotionFerPlus;              // [35] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus
  class LITE_EXPORTS TNNSSRNet;                      // [36] * reference: https://github.com/oukohou/SSR_Net_Pytorch
  class LITE_EXPORTS TNNEfficientEmotion7;           // [37] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS TNNEfficientEmotion8;           // [38] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS TNNMobileEmotion7;              // [39] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS TNNReXNetEmotion7;              // [40] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS TNNEfficientNetLite4;           // [41] * reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class LITE_EXPORTS TNNShuffleNetV2;                // [42] * reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class LITE_EXPORTS TNNDenseNet;                    // [43] * reference: https://pytorch.org/hub/pytorch_vision_densenet/
  class LITE_EXPORTS TNNGhostNet;                    // [44] * reference：https://pytorch.org/hub/pytorch_vision_ghostnet/
  class LITE_EXPORTS TNNHdrDNet;                     // [45] * reference: https://pytorch.org/hub/pytorch_vision_hardnet/
  class LITE_EXPORTS TNNIBNNet;                      // [46] * reference: https://pytorch.org/hub/pytorch_vision_ibnnet/
  class LITE_EXPORTS TNNMobileNetV2;                 // [47] * reference: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
  class LITE_EXPORTS TNNResNet;                      // [48] * reference: https://pytorch.org/hub/pytorch_vision_resnet/
  class LITE_EXPORTS TNNResNeXt;                     // [49] * reference: https://pytorch.org/hub/pytorch_vision_resnext/
  class LITE_EXPORTS TNNFastStyleTransfer;           // [50] * reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class LITE_EXPORTS TNNColorizer;                   // [51] * reference: https://github.com/richzhang/colorization
  class LITE_EXPORTS TNNSubPixelCNN;                 // [52] * reference: https://github.com/niazwazir/SUB_PIXEL_CNN
  class LITE_EXPORTS TNNDeepLabV3ResNet101;          // [53] * reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  class LITE_EXPORTS TNNFCNResNet101;                // [54] * reference: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
  class LITE_EXPORTS TNNMGMatting;                   // [55] * reference: https://github.com/yucornetto/MGMatting
  class LITE_EXPORTS TNNNanoDetPlus;                 // [56] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS TNNSCRFD;                       // [57] * reference: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
  class LITE_EXPORTS TNNYOLO5Face;                   // [58] * reference: https://github.com/deepcam-cn/yolov5-face

}

namespace tnncv
{
  using tnncore::BasicTNNHandler;
}

namespace tnnnlp
{
  using tnncore::BasicTNNHandler;
}

namespace tnnasr
{
  using tnncore::BasicTNNHandler;
}

#endif //LITE_AI_TOOLKIT_TNN_CORE_TNN_CORE_H
