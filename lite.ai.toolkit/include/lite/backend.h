//
// Created by DefTruth on 2021/5/30.
//

#ifndef LITE_AI_BACKEND_H
#define LITE_AI_BACKEND_H

#include "config.h"

// BACKEND ONNXRuntime
#ifdef BACKEND_ONNXRUNTIME

# ifdef BACKEND_NCNN
# undef BACKEND_NCNN
# endif

# ifdef BACKEND_MNN
# undef BACKEND_MNN
# endif

# ifdef BACKEND_TNN
# undef BACKEND_TNN
# endif

#endif

// BACKEND NCNN
#ifdef BACKEND_NCNN

# ifdef BACKEND_ONNXRUNTIME
# undef BACKEND_ONNXRUNTIME
# endif

# ifdef BACKEND_MNN
# undef BACKEND_MNN
# endif

# ifdef BACKEND_TNN
# undef BACKEND_TNN
# endif

#endif

// BACKEND MNN
#ifdef BACKEND_MNN

# ifdef BACKEND_NCNN
# undef BACKEND_NCNN
# endif

# ifdef BACKEND_ONNXRUNTIME
# undef BACKEND_ONNXRUNTIME
# endif

# ifdef BACKEND_TNN
# undef BACKEND_TNN
# endif

#endif

// BACKEND TNN
#ifdef BACKEND_TNN

# ifdef BACKEND_NCNN
# undef BACKEND_NCNN
# endif

# ifdef BACKEND_MNN
# undef BACKEND_MNN
# endif

# ifdef BACKEND_ONNXRUNTIME
# undef BACKEND_ONNXRUNTIME
# endif

#endif

#endif //LITE_AI_BACKEND_H
