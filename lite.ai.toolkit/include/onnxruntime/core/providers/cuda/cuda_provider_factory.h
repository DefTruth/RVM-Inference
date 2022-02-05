// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include "onnxruntime_c_api.h" // The official code of onnxruntime 1.7.0
// The new include line was update by lite.ai.toolkit at 2021107 to make
// sure the 'onnxruntime_c_api.h' header can be include into this file
// correctly.
#include "onnxruntime/core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_id cuda device id, starts from zero.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);

#ifdef __cplusplus
}
#endif
