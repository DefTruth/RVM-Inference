//
// Created by DefTruth on 2021/8/8.
//

#ifndef LITE_AI_MODELS_H
#define LITE_AI_MODELS_H

#include "backend.h"

// ENABLE_ONNXRUNTIME
#ifdef ENABLE_ONNXRUNTIME

#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/ort/cv/age_googlenet.h"
#include "lite/ort/cv/glint_arcface.h"
#include "lite/ort/cv/colorizer.h"
#include "lite/ort/cv/deeplabv3_resnet101.h"
#include "lite/ort/cv/densenet.h"
#include "lite/ort/cv/efficientnet_lite4.h"
#include "lite/ort/cv/emotion_ferplus.h"
#include "lite/ort/cv/fast_style_transfer.h"
#include "lite/ort/cv/fcn_resnet101.h"
#include "lite/ort/cv/fsanet.h"
#include "lite/ort/cv/gender_googlenet.h"
#include "lite/ort/cv/ghostnet.h"
#include "lite/ort/cv/hardnet.h"
#include "lite/ort/cv/ibnnet.h"
#include "lite/ort/cv/mobilenetv2.h"
#include "lite/ort/cv/pfld.h"
#include "lite/ort/cv/resnet.h"
#include "lite/ort/cv/resnext.h"
#include "lite/ort/cv/shufflenetv2.h"
#include "lite/ort/cv/ssd.h"
#include "lite/ort/cv/ssd_mobilenetv1.h"
#include "lite/ort/cv/ssrnet.h"
#include "lite/ort/cv/subpixel_cnn.h"
#include "lite/ort/cv/tiny_yolov3.h"
#include "lite/ort/cv/ultraface.h"
#include "lite/ort/cv/vgg16_age.h"
#include "lite/ort/cv/vgg16_gender.h"
#include "lite/ort/cv/yolov3.h"
#include "lite/ort/cv/yolov4.h"
#include "lite/ort/cv/yolov5.h"
#include "lite/ort/cv/glint_cosface.h"
#include "lite/ort/cv/glint_partial_fc.h"
#include "lite/ort/cv/facenet.h"
#include "lite/ort/cv/focal_arcface.h"
#include "lite/ort/cv/focal_asia_arcface.h"
#include "lite/ort/cv/tencent_cifp_face.h"
#include "lite/ort/cv/tencent_curricular_face.h"
#include "lite/ort/cv/center_loss_face.h"
#include "lite/ort/cv/sphere_face.h"
#include "lite/ort/cv/pose_robust_face.h"
#include "lite/ort/cv/naive_pose_robust_face.h"
#include "lite/ort/cv/mobile_facenet.h"
#include "lite/ort/cv/cava_ghost_arcface.h"
#include "lite/ort/cv/cava_combined_face.h"
#include "lite/ort/cv/yolox.h"
#include "lite/ort/cv/mobilese_focal_face.h"
#include "lite/ort/cv/efficient_emotion7.h"
#include "lite/ort/cv/efficient_emotion8.h"
#include "lite/ort/cv/mobile_emotion7.h"
#include "lite/ort/cv/rexnet_emotion7.h"
#include "lite/ort/cv/pfld98.h"
#include "lite/ort/cv/pfld68.h"
#include "lite/ort/cv/mobilenetv2_68.h"
#include "lite/ort/cv/mobilenetv2_se_68.h"
#include "lite/ort/cv/face_landmarks_1000.h"
#include "lite/ort/cv/retinaface.h"
#include "lite/ort/cv/faceboxes.h"
#include "lite/ort/cv/tiny_yolov4_voc.h"
#include "lite/ort/cv/tiny_yolov4_coco.h"
#include "lite/ort/cv/yolor.h"
#include "lite/ort/cv/scaled_yolov4.h"
#include "lite/ort/cv/efficientdet.h"
#include "lite/ort/cv/efficientdet_d7.h"
#include "lite/ort/cv/efficientdet_d8.h"
#include "lite/ort/cv/yolop.h"
#include "lite/ort/cv/rvm.h"
#include "lite/ort/cv/nanodet.h"
#include "lite/ort/cv/nanodet_efficientnet_lite.h"

#endif

// ENABLE_MNN
#ifdef ENABLE_MNN

#include "lite/mnn/core/mnn_core.h"
#include "lite/mnn/core/mnn_utils.h"
#include "lite/mnn/cv/mnn_nanodet.h"
#include "lite/mnn/cv/mnn_nanodet_efficientnet_lite.h"
#include "lite/mnn/cv/mnn_rvm.h"
#include "lite/mnn/cv/mnn_yolox.h"

#endif

// ENABLE_NCNN
#ifdef ENABLE_NCNN

#include "lite/ncnn/core/ncnn_core.h"
#include "lite/ncnn/core/ncnn_utils.h"
#include "lite/ncnn/cv/ncnn_nanodet.h"
#include "lite/ncnn/cv/ncnn_nanodet_efficientnet_lite.h"
#include "lite/ncnn/cv/ncnn_nanodet_depreciated.h"
#include "lite/ncnn/cv/ncnn_nanodet_efficientdet_lite_depreciated.h"
#include "lite/ncnn/cv/ncnn_rvm.h"

#endif

// ENABLE_TNN
#ifdef ENABLE_TNN
#endif

// Default Engine ONNXRuntime
namespace lite
{
  namespace cv
  {
#ifdef BACKEND_ONNXRUNTIME
    typedef ortcv::FSANet _FSANet;
    typedef ortcv::PFLD _PFLD;
    typedef ortcv::UltraFace _UltraFace;
    typedef ortcv::AgeGoogleNet _AgeGoogleNet;
    typedef ortcv::GenderGoogleNet _GenderGoogleNet;
    typedef ortcv::EmotionFerPlus _EmotionFerPlus;
    typedef ortcv::VGG16Age _VGG16Age;
    typedef ortcv::VGG16Gender _VGG16Gender;
    typedef ortcv::SSRNet _SSRNet;
    typedef ortcv::FastStyleTransfer _FastStyleTransfer;
    typedef ortcv::GlintArcFace _GlintArcFace;
    typedef ortcv::Colorizer _Colorizer;
    typedef ortcv::SubPixelCNN _SubPixelCNN;
    typedef ortcv::YoloV4 _YoloV4;
    typedef ortcv::YoloV3 _YoloV3;
    typedef ortcv::YoloV5 _YoloV5;
    typedef ortcv::EfficientNetLite4 _EfficientNetLite4;
    typedef ortcv::ShuffleNetV2 _ShuffleNetV2;
    typedef ortcv::TinyYoloV3 _TinyYoloV3;
    typedef ortcv::SSD _SSD;
    typedef ortcv::SSDMobileNetV1 _SSDMobileNetV1;
    typedef ortcv::DeepLabV3ResNet101 _DeepLabV3ResNet101;
    typedef ortcv::DenseNet _DenseNet;
    typedef ortcv::FCNResNet101 _FCNResNet101;
    typedef ortcv::GhostNet _GhostNet;
    typedef ortcv::HdrDNet _HdrDNet;
    typedef ortcv::IBNNet _IBNNet;
    typedef ortcv::MobileNetV2 _MobileNetV2;
    typedef ortcv::ResNet _ResNet;
    typedef ortcv::ResNeXt _ResNeXt;
    typedef ortcv::GlintCosFace _GlintCosFace;
    typedef ortcv::GlintPartialFC _GlintPartialFC;
    typedef ortcv::FaceNet _FaceNet;
    typedef ortcv::FocalArcFace _FocalArcFace;
    typedef ortcv::FocalAsiaArcFace _FocalAsiaArcFace;
    typedef ortcv::TencentCifpFace _TencentCifpFace;
    typedef ortcv::TencentCurricularFace _TencentCurricularFace;
    typedef ortcv::CenterLossFace _CenterLossFace;
    typedef ortcv::SphereFace _SphereFace;
    typedef ortcv::PoseRobustFace _PoseRobustFace;
    typedef ortcv::NaivePoseRobustFace _NaivePoseRobustFace;
    typedef ortcv::MobileFaceNet _MobileFaceNet;
    typedef ortcv::CavaGhostArcFace _CavaGhostArcFace;
    typedef ortcv::CavaCombinedFace _CavaCombinedFace;
    typedef ortcv::YoloX _YoloX;
    typedef ortcv::MobileSEFocalFace _MobileSEFocalFace;
    typedef ortcv::EfficientEmotion7 _EfficientEmotion7;
    typedef ortcv::EfficientEmotion8 _EfficientEmotion8;
    typedef ortcv::MobileEmotion7 _MobileEmotion7;
    typedef ortcv::ReXNetEmotion7 _ReXNetEmotion7;
    typedef ortcv::PFLD98 _PFLD98;
    typedef ortcv::PFLD68 _PFLD68;
    typedef ortcv::MobileNetV268 _MobileNetV268;
    typedef ortcv::MobileNetV2SE68 _MobileNetV2SE68;
    typedef ortcv::FaceLandmark1000 _FaceLandmark1000;
    typedef ortcv::RetinaFace _RetinaFace;
    typedef ortcv::FaceBoxes _FaceBoxes;
    typedef ortcv::TinyYoloV4VOC _TinyYoloV4VOC;
    typedef ortcv::TinyYoloV4COCO _TinyYoloV4COCO;
    typedef ortcv::YoloR _YoloR;
    typedef ortcv::ScaledYoloV4 _ScaledYoloV4;
    typedef ortcv::EfficientDet _EfficientDet;
    typedef ortcv::EfficientDetD7 _EfficientDetD7;
    typedef ortcv::EfficientDetD8 _EfficientDetD8;
    typedef ortcv::YOLOP _YOLOP;
    typedef ortcv::RobustVideoMatting _RobustVideoMatting;
    typedef ortcv::NanoDet _NanoDet;
    typedef ortcv::NanoDetEfficientNetLite _NanoDetEfficientNetLite;
#endif

    // 1. classification
    namespace classification
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _EfficientNetLite4 EfficientNetLite4;
      typedef _ShuffleNetV2 ShuffleNetV2;
      typedef _DenseNet DenseNet;
      typedef _GhostNet GhostNet;
      typedef _HdrDNet HdrDNet;
      typedef _IBNNet IBNNet;
      typedef _MobileNetV2 MobileNetV2;
      typedef _ResNet ResNet;
      typedef _ResNeXt ResNeXt;
#endif
    }

    // 2. general object detection
    namespace detection
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _YoloV3 YoloV3;
      typedef _YoloV4 YoloV4;
      typedef _YoloV5 YoloV5;
      typedef _TinyYoloV3 TinyYoloV3;
      typedef _SSD SSD;
      typedef _SSDMobileNetV1 SSDMobileNetV1;
      typedef _YoloX YoloX;
      typedef _TinyYoloV4VOC TinyYoloV4VOC;
      typedef _TinyYoloV4COCO TinyYoloV4COCO;
      typedef _YoloR YoloR;
      typedef _ScaledYoloV4 ScaledYoloV4;
      typedef _EfficientDet EfficientDet;
      typedef _EfficientDetD7 EfficientDetD7;
      typedef _EfficientDetD8 EfficientDetD8;
      typedef _YOLOP YOLOP;
      typedef _NanoDet NanoDet;
      typedef _NanoDetEfficientNetLite NanoDetEfficientNetLite;
#endif
    }
    // 3. face detection & facial attributes detection
    namespace face
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _FSANet FSANet; // head pose estimation.
      typedef _UltraFace UltraFace;  // face detection.
      typedef _PFLD PFLD; // facial landmarks detection.
      typedef _AgeGoogleNet AgeGoogleNet; // age estimation
      typedef _GenderGoogleNet GenderGoogleNet; // gender estimation
      typedef _VGG16Age VGG16Age; // age estimation
      typedef _VGG16Gender VGG16Gender; // gender estimation
      typedef _EmotionFerPlus EmotionFerPlus; // emotion detection
      typedef _SSRNet SSRNet; // age estimation
      typedef _EfficientEmotion7 EfficientEmotion7;
      typedef _EfficientEmotion8 EfficientEmotion8;
      typedef _MobileEmotion7 MobileEmotion7;
      typedef _ReXNetEmotion7 ReXNetEmotion7;
      typedef _PFLD98 PFLD98;
      typedef _PFLD68 PFLD68;
      typedef _MobileNetV268 MobileNetV268;
      typedef _MobileNetV2SE68 MobileNetV2SE68;
      typedef _FaceLandmark1000 FaceLandmark1000;
      typedef _RetinaFace RetinaFace;
      typedef _FaceBoxes FaceBoxes;
#endif
      namespace detect
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _UltraFace UltraFace;  // face detection.
        typedef _RetinaFace RetinaFace;
        typedef _FaceBoxes FaceBoxes;
#endif
      }

      namespace align
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _PFLD PFLD; // facial landmarks detection. 106 points
        typedef _PFLD98 PFLD98; // 98 points
        typedef _PFLD68 PFLD68; // 68 points
        typedef _MobileNetV268 MobileNetV268; // 68 points
        typedef _MobileNetV2SE68 MobileNetV2SE68; // 68 points
        typedef _FaceLandmark1000 FaceLandmark1000; // 1000 points
#endif
      }

      namespace pose
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _FSANet FSANet; // head pose estimation.
#endif
      }

      namespace attr
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _AgeGoogleNet AgeGoogleNet; // age estimation
        typedef _GenderGoogleNet GenderGoogleNet; // gender estimation
        typedef _VGG16Age VGG16Age; // age estimation
        typedef _VGG16Gender VGG16Gender; // gender estimation
        typedef _EmotionFerPlus EmotionFerPlus; // emotion detection
        typedef _SSRNet SSRNet; // age estimation
        typedef _EfficientEmotion7 EfficientEmotion7;
        typedef _EfficientEmotion8 EfficientEmotion8;
        typedef _MobileEmotion7 MobileEmotion7;
        typedef _ReXNetEmotion7 ReXNetEmotion7;
#endif
      }

    }
    // 4. face recognition
    namespace faceid
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _GlintArcFace GlintArcFace; //
      typedef _GlintCosFace GlintCosFace; //
      typedef _GlintPartialFC GlintPartialFC;
      typedef _FaceNet FaceNet;
      typedef _FocalArcFace FocalArcFace;
      typedef _FocalAsiaArcFace FocalAsiaArcFace;
      typedef _TencentCurricularFace TencentCurricularFace;
      typedef _TencentCifpFace TencentCifpFace;
      typedef _CenterLossFace CenterLossFace;
      typedef _SphereFace SphereFace;
      typedef _PoseRobustFace PoseRobustFace;
      typedef _NaivePoseRobustFace NaivePoseRobustFace;
      typedef _MobileFaceNet MobileFaceNet;
      typedef _CavaGhostArcFace CavaGhostArcFace;
      typedef _CavaCombinedFace CavaCombinedFace;
      typedef _MobileSEFocalFace MobileSEFocalFace;
#endif

    }
    // 5. segmentation
    namespace segmentation
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _DeepLabV3ResNet101 DeepLabV3ResNet101;
      typedef _FCNResNet101 FCNResNet101;
#endif

    }
    // 6. reid
    namespace reid
    {
#ifdef BACKEND_ONNXRUNTIME
#endif
    }

    // 7. ocr
    namespace ocr
    {
#ifdef BACKEND_ONNXRUNTIME
#endif
    }
    // 8. neural rendering
    namespace render
    {
#ifdef BACKEND_ONNXRUNTIME
#endif
    }
    // 9. style transfer
    namespace style
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _FastStyleTransfer FastStyleTransfer;
#endif
    }

    // 10. colorization
    namespace colorization
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _Colorizer Colorizer;
#endif
    }
    // 11. super resolution
    namespace resolution
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _SubPixelCNN SubPixelCNN;
#endif
    }
    // 12. image & face & human matting
    namespace matting
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _RobustVideoMatting RobustVideoMatting;
#endif
    }
  }

  namespace asr
  {
#ifdef BACKEND_ONNXRUNTIME
#endif
  }

  namespace nlp
  {
#ifdef BACKEND_ONNXRUNTIME
#endif
  }
}

// models for mobile device
namespace lite
{
  namespace mobile
  {
    // classification
    namespace classification
    {
    }
    // object detection
    namespace detection
    {
    }
    // face etc.
    namespace face
    {
      namespace detect
      {
      }
      namespace align
      {
      }
      namespace pose
      {
      }
      namespace attr
      {
      }
    }
    // face recognition
    namespace faceid
    {
    }
    // segmentation
    namespace segmentation
    {
    }
    // reid
    namespace reid
    {
    }
    // ocr
    namespace ocr
    {
    }
    // matting
    namespace matting
    {
    }

  }
}

// ONNXRuntime version
namespace lite
{
#ifdef ENABLE_ONNXRUNTIME
  namespace onnxruntime
  {
    namespace cv
    {
      typedef ortcv::FSANet _ONNXFSANet;
      typedef ortcv::PFLD _ONNXPFLD;
      typedef ortcv::UltraFace _ONNXUltraFace;
      typedef ortcv::AgeGoogleNet _ONNXAgeGoogleNet;
      typedef ortcv::GenderGoogleNet _ONNXGenderGoogleNet;
      typedef ortcv::EmotionFerPlus _ONNXEmotionFerPlus;
      typedef ortcv::VGG16Age _ONNXVGG16Age;
      typedef ortcv::VGG16Gender _ONNXVGG16Gender;
      typedef ortcv::SSRNet _ONNXSSRNet;
      typedef ortcv::FastStyleTransfer _ONNXFastStyleTransfer;
      typedef ortcv::GlintArcFace _ONNXGlintArcFace;
      typedef ortcv::Colorizer _ONNXColorizer;
      typedef ortcv::SubPixelCNN _ONNXSubPixelCNN;
      typedef ortcv::YoloV4 _ONNXYoloV4;
      typedef ortcv::YoloV3 _ONNXYoloV3;
      typedef ortcv::YoloV5 _ONNXYoloV5;
      typedef ortcv::EfficientNetLite4 _ONNXEfficientNetLite4;
      typedef ortcv::ShuffleNetV2 _ONNXShuffleNetV2;
      typedef ortcv::TinyYoloV3 _ONNXTinyYoloV3;
      typedef ortcv::SSD _ONNXSSD;
      typedef ortcv::SSDMobileNetV1 _ONNXSSDMobileNetV1;
      typedef ortcv::DeepLabV3ResNet101 _ONNXDeepLabV3ResNet101;
      typedef ortcv::DenseNet _ONNXDenseNet;
      typedef ortcv::FCNResNet101 _ONNXFCNResNet101;
      typedef ortcv::GhostNet _ONNXGhostNet;
      typedef ortcv::HdrDNet _ONNXHdrDNet;
      typedef ortcv::IBNNet _ONNXIBNNet;
      typedef ortcv::MobileNetV2 _ONNXMobileNetV2;
      typedef ortcv::ResNet _ONNXResNet;
      typedef ortcv::ResNeXt _ONNXResNeXt;
      typedef ortcv::GlintCosFace _ONNXGlintCosFace;
      typedef ortcv::GlintPartialFC _ONNXGlintPartialFC;
      typedef ortcv::FaceNet _ONNXFaceNet;
      typedef ortcv::FocalArcFace _ONNXFocalArcFace;
      typedef ortcv::FocalAsiaArcFace _ONNXFocalAsiaArcFace;
      typedef ortcv::TencentCifpFace _ONNXTencentCifpFace;
      typedef ortcv::TencentCurricularFace _ONNXTencentCurricularFace;
      typedef ortcv::CenterLossFace _ONNXCenterLossFace;
      typedef ortcv::SphereFace _ONNXSphereFace;
      typedef ortcv::PoseRobustFace _ONNXPoseRobustFace;
      typedef ortcv::NaivePoseRobustFace _ONNXNaivePoseRobustFace;
      typedef ortcv::MobileFaceNet _ONNXMobileFaceNet;
      typedef ortcv::CavaGhostArcFace _ONNXCavaGhostArcFace;
      typedef ortcv::CavaCombinedFace _ONNXCavaCombinedFace;
      typedef ortcv::YoloX _ONNXYoloX;
      typedef ortcv::MobileSEFocalFace _ONNXMobileSEFocalFace;
      typedef ortcv::EfficientEmotion7 _ONNXEfficientEmotion7;
      typedef ortcv::EfficientEmotion8 _ONNXEfficientEmotion8;
      typedef ortcv::MobileEmotion7 _ONNXMobileEmotion7;
      typedef ortcv::ReXNetEmotion7 _ONNXReXNetEmotion7;
      typedef ortcv::PFLD98 _ONNXPFLD98;
      typedef ortcv::PFLD68 _ONNXPFLD68;
      typedef ortcv::MobileNetV268 _ONNXMobileNetV268;
      typedef ortcv::MobileNetV2SE68 _ONNXMobileNetV2SE68;
      typedef ortcv::FaceLandmark1000 _ONNXFaceLandmark1000;
      typedef ortcv::RetinaFace _ONNXRetinaFace;
      typedef ortcv::FaceBoxes _ONNXFaceBoxes;
      typedef ortcv::TinyYoloV4VOC _ONNXTinyYoloV4VOC;
      typedef ortcv::TinyYoloV4COCO _ONNXTinyYoloV4COCO;
      typedef ortcv::YoloR _ONNXYoloR;
      typedef ortcv::ScaledYoloV4 _ONNXScaledYoloV4;
      typedef ortcv::EfficientDet _ONNXEfficientDet;
      typedef ortcv::EfficientDetD7 _ONNXEfficientDetD7;
      typedef ortcv::EfficientDetD8 _ONNXEfficientDetD8;
      typedef ortcv::YOLOP _ONNXYOLOP;
      typedef ortcv::RobustVideoMatting _ONNXRobustVideoMatting;
      typedef ortcv::NanoDet _ONNXNanoDet;
      typedef ortcv::NanoDetEfficientNetLite _ONNXNanoDetEfficientNetLite;


      // 1. classification
      namespace classification
      {
        typedef _ONNXEfficientNetLite4 EfficientNetLite4;
        typedef _ONNXShuffleNetV2 ShuffleNetV2;
        typedef _ONNXDenseNet DenseNet;
        typedef _ONNXGhostNet GhostNet;
        typedef _ONNXHdrDNet HdrDNet;
        typedef _ONNXIBNNet IBNNet;
        typedef _ONNXMobileNetV2 MobileNetV2;
        typedef _ONNXResNet ResNet;
        typedef _ONNXResNeXt ResNeXt;
      }

      // 2. general object detection
      namespace detection
      {
        typedef _ONNXYoloV3 YoloV3;
        typedef _ONNXYoloV4 YoloV4;
        typedef _ONNXYoloV5 YoloV5;
        typedef _ONNXTinyYoloV3 TinyYoloV3;
        typedef _ONNXSSD SSD;
        typedef _ONNXSSDMobileNetV1 SSDMobileNetV1;
        typedef _ONNXYoloX YoloX;
        typedef _ONNXTinyYoloV4VOC TinyYoloV4VOC;
        typedef _ONNXTinyYoloV4COCO TinyYoloV4COCO;
        typedef _ONNXYoloR YoloR;
        typedef _ONNXScaledYoloV4 ScaledYoloV4;
        typedef _ONNXEfficientDet EfficientDet;
        typedef _ONNXEfficientDetD7 EfficientDetD7;
        typedef _ONNXEfficientDetD8 EfficientDetD8;
        typedef _ONNXYOLOP YOLOP;
        typedef _ONNXNanoDet NanoDet;
        typedef _ONNXNanoDetEfficientNetLite NanoDetEfficientNetLite;
      }
      // 3. face detection & facial attributes detection
      namespace face
      {
        typedef _ONNXFSANet FSANet; // head pose estimation.
        typedef _ONNXUltraFace UltraFace;  // face detection.
        typedef _ONNXPFLD PFLD; // facial landmarks detection.
        typedef _ONNXAgeGoogleNet AgeGoogleNet; // age estimation
        typedef _ONNXGenderGoogleNet GenderGoogleNet; // gender estimation
        typedef _ONNXVGG16Age VGG16Age; // age estimation
        typedef _ONNXVGG16Gender VGG16Gender; // gender estimation
        typedef _ONNXEmotionFerPlus EmotionFerPlus; // emotion detection
        typedef _ONNXSSRNet SSRNet; // age estimation
        typedef _ONNXEfficientEmotion7 EfficientEmotion7;
        typedef _ONNXEfficientEmotion8 EfficientEmotion8;
        typedef _ONNXMobileEmotion7 MobileEmotion7;
        typedef _ONNXReXNetEmotion7 ReXNetEmotion7;
        typedef _ONNXPFLD98 PFLD98;
        typedef _ONNXPFLD68 PFLD68;
        typedef _ONNXMobileNetV268 MobileNetV268;
        typedef _ONNXMobileNetV2SE68 MobileNetV2SE68;
        typedef _ONNXFaceLandmark1000 FaceLandmark1000;
        typedef _ONNXRetinaFace RetinaFace;
        typedef _ONNXFaceBoxes FaceBoxes;

        namespace detect
        {
          typedef _ONNXUltraFace UltraFace;  // face detection.
          typedef _ONNXRetinaFace RetinaFace;
          typedef _ONNXFaceBoxes FaceBoxes;
        }

        namespace align
        {
          typedef _ONNXPFLD PFLD; // facial landmarks detection. 106 points
          typedef _ONNXPFLD98 PFLD98; // 98 points
          typedef _ONNXPFLD68 PFLD68; // 68 points
          typedef _ONNXMobileNetV268 MobileNetV268; // 68 points
          typedef _ONNXMobileNetV2SE68 MobileNetV2SE68; // 68 points
          typedef _ONNXFaceLandmark1000 FaceLandmark1000; // 1000 points
        }

        namespace pose
        {
          typedef _ONNXFSANet FSANet; // head pose estimation.
        }

        namespace attr
        {
          typedef _ONNXAgeGoogleNet AgeGoogleNet; // age estimation
          typedef _ONNXGenderGoogleNet GenderGoogleNet; // gender estimation
          typedef _ONNXVGG16Age VGG16Age; // age estimation
          typedef _ONNXVGG16Gender VGG16Gender; // gender estimation
          typedef _ONNXEmotionFerPlus EmotionFerPlus; // emotion detection
          typedef _ONNXSSRNet SSRNet; // age estimation
          typedef _ONNXEfficientEmotion7 EfficientEmotion7;
          typedef _ONNXEfficientEmotion8 EfficientEmotion8;
          typedef _ONNXMobileEmotion7 MobileEmotion7;
          typedef _ONNXReXNetEmotion7 ReXNetEmotion7;
        }
      }
      // 4. face recognition
      namespace faceid
      {
        typedef _ONNXGlintArcFace GlintArcFace; //
        typedef _ONNXGlintCosFace GlintCosFace; //
        typedef _ONNXGlintPartialFC GlintPartialFC;
        typedef _ONNXFaceNet FaceNet;
        typedef _ONNXFocalArcFace FocalArcFace;
        typedef _ONNXFocalAsiaArcFace FocalAsiaArcFace;
        typedef _ONNXTencentCifpFace TencentCifpFace;
        typedef _ONNXTencentCurricularFace TencentCurricularFace;
        typedef _ONNXCenterLossFace CenterLossFace;
        typedef _ONNXSphereFace SphereFace;
        typedef _ONNXPoseRobustFace PoseRobustFace;
        typedef _ONNXNaivePoseRobustFace NaivePoseRobustFace;
        typedef _ONNXMobileFaceNet MobileFaceNet;
        typedef _ONNXCavaGhostArcFace CavaGhostArcFace;
        typedef _ONNXCavaCombinedFace CavaCombinedFace;
        typedef _ONNXMobileSEFocalFace MobileSEFocalFace;

      }
      // 5. segmentation
      namespace segmentation
      {
        typedef _ONNXDeepLabV3ResNet101 DeepLabV3ResNet101;
        typedef _ONNXFCNResNet101 FCNResNet101;

      }
      // 6. reid
      namespace reid
      {

      }

      // 7. ocr
      namespace ocr
      {

      }
      // 8. neural rendering
      namespace render
      {

      }
      // 9. style transfer
      namespace style
      {
        typedef _ONNXFastStyleTransfer FastStyleTransfer;
      }

      // 10. colorization
      namespace colorization
      {
        typedef _ONNXColorizer Colorizer;
      }
      // 11. super resolution
      namespace resolution
      {
        typedef _ONNXSubPixelCNN SubPixelCNN;
      }
      // 12. image & face & human matting
      namespace matting
      {
        typedef _ONNXRobustVideoMatting RobustVideoMatting;
      }

    }

  }
#endif
}

// MNN version
namespace lite
{
#ifdef ENABLE_MNN
  namespace mnn
  {
    namespace cv
    {
      // classification
      namespace classification
      {
      }
      // object detection
      namespace detection
      {
        typedef mnncv::MNNNanoDet NanoDet;
        typedef mnncv::MNNNanoDetEfficientNetLite NanoDetEfficientNetLite;
        typedef mnncv::MNNYoloX YoloX;

      }
      // face etc.
      namespace face
      {
        namespace detect
        {
        }
        namespace align
        {
        }
        namespace pose
        {
        }
        namespace attr
        {
        }
      }
      // face recognition
      namespace faceid
      {
      }
      // segmentation
      namespace segmentation
      {
      }
      // reid
      namespace reid
      {
      }
      // ocr
      namespace ocr
      {
      }
      // matting
      namespace matting
      {
        typedef mnncv::MNNRobustVideoMatting RobustVideoMatting;
      }

    } // namespace cv

  }
#endif
}

// NCNN version
namespace lite
{
#ifdef ENABLE_NCNN
  namespace ncnn
    {
    namespace cv
    {
      // classification
      namespace classification
      {
      }
      // object detection
      namespace detection
      {
        typedef ncnncv::NCNNNanoDet NanoDet;
        typedef ncnncv::NCNNNanoDetEfficientNetLite NanoDetEfficientNetLite;
        typedef ncnncv::NCNNNanoDetDepreciated NanoDetDepreciated;
        typedef ncnncv::NCNNNanoDetEfficientNetLiteDepreciated NanoDetEfficientNetLiteDepreciated;
      }
      // face etc.
      namespace face
      {
        namespace detect
        {
        }
        namespace align
        {
        }
        namespace pose
        {
        }
        namespace attr
        {
        }
      }
      // face recognition
      namespace faceid
      {
      }
      // segmentation
      namespace segmentation
      {
      }
      // reid
      namespace reid
      {
      }
      // ocr
      namespace ocr
      {
      }
      // matting
      namespace matting
      {
        typedef ncnncv::NCNNRobustVideoMatting RobustVideoMatting;
      }

    } // namespace cv

  }
#endif
}

// TNN version
namespace lite
{
#ifdef ENABLE_TNN
  namespace tnn
    {
    }
#endif
}

#endif //LITE_AI_MODELS_H
