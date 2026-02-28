#include <stdexcept>
#include <chrono>
#include "YoloTask.h"
#include "YoloOpenCVRT.h"
#ifdef BUILD_WITH_ORT
#include "ort/YoloONNXRT.h"
#endif
#ifdef BUILD_WITH_OVN
#include "ovn/YoloOVNRT.h"
#endif
#ifdef BUILD_WITH_TRT
#include "trt/YoloTRT.h"
#endif
#ifdef BUILD_WITH_RKN
#include "rkn/YoloRKNNRT.h"
#endif
namespace yolo {
        
    YoloTask::YoloTask(const YoloConfig& cfg): _cfg(cfg) {
        switch (_cfg.target_rt) {
        case YoloTargetRT::OPENCV_CPU:
            _rt = std::make_shared<yolo::YoloOpenCVRT>(_cfg.model_path, false);
            break;
        case YoloTargetRT::OPENCV_CUDA:
            _rt = std::make_shared<yolo::YoloOpenCVRT>(_cfg.model_path, true);
            break;
        #ifdef BUILD_WITH_ORT
        case YoloTargetRT::ORT_CPU:
            _rt = std::make_shared<yolo::YoloONNXRT>(_cfg.model_path, false);
            break;
        case YoloTargetRT::ORT_CUDA:
            _rt = std::make_shared<yolo::YoloONNXRT>(_cfg.model_path, true);
            break;
        #endif
        #ifdef BUILD_WITH_OVN
        case YoloTargetRT::OVN_AUTO:
            _rt = std::make_shared<yolo::YoloOVNRT>(_cfg.model_path, "AUTO");
            break;
        case YoloTargetRT::OVN_CPU:
            _rt = std::make_shared<yolo::YoloOVNRT>(_cfg.model_path, "CPU");
            break;
        case YoloTargetRT::OVN_GPU:
            _rt = std::make_shared<yolo::YoloOVNRT>(_cfg.model_path, "GPU");
            break;
        #endif
        #ifdef BUILD_WITH_TRT
        case YoloTargetRT::TRT:
            _rt = std::make_shared<yolo::YoloTRT>(_cfg.model_path);
            break;
        #endif
        #ifdef BUILD_WITH_RKN
        case YoloTargetRT::RKNN:
            _rt = std::make_shared<yolo::YoloRKNNRT>(_cfg.model_path);
            break;
        #endif
        default:
            throw std::invalid_argument("invalid(unsupported) YoloTargetRT parameter when initializing YoloTask!");
            break;
        }
    }
    
    YoloTask::~YoloTask() {

    }
    
    cv::Mat YoloTask::preprocess_one(const cv::Mat& image) {
        assert(!image.empty());

        // different for classification task
        if (_cfg.task == YoloTaskType::CLS) {
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(_cfg.input_w, _cfg.input_h));
            _orig_sizes.push_back(image.size());
            _letterbox_infos.push_back(LetterBoxInfo{1.0f, 0, 0});
            _input_images.push_back(resized);

            // [3, _cfg.input_h, _cfg.input_w]
            return resized;
        }
        
        LetterBoxInfo info;
        YoloUtils utils;
        cv::Mat lb = utils.letterbox(
            image,
            _cfg.input_w,
            _cfg.input_h,
            info
        );
        _orig_sizes.push_back(image.size());
        _letterbox_infos.push_back(info);
        _input_images.push_back(lb);
        // [3, _cfg.input_h, _cfg.input_w]
        return lb;
    }

    cv::Mat YoloTask::preprocess(const std::vector<cv::Mat>& images) {
        _orig_sizes.clear();
        _letterbox_infos.clear();
        _input_images.clear();

        std::vector<cv::Mat> letterboxes;
        for (const auto& image: images) {
            letterboxes.push_back(preprocess_one(image));
        }

        // [batch, 3, _cfg.input_h, _cfg.input_w]
        cv::Mat blob;
        cv::dnn::blobFromImages(
            letterboxes,
            blob,
            _cfg.scale_f,
            cv::Size(),
            cv::Scalar(),
            _cfg.rgb,
            false
        );

        // different for classification task
        if (_cfg.task == YoloTaskType::CLS
            && _cfg.mean.size() == 3 
            && _cfg.std.size() == 3) {
            int N = blob.size[0];
            int C = blob.size[1];
            int H = blob.size[2];
            int W = blob.size[3];

            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    float* ptr = blob.ptr<float>(n, c);
                    float m = _cfg.mean[c];
                    float s = _cfg.std[c];

                    int spatial = H * W;
                    for (int i = 0; i < spatial; ++i) {
                        ptr[i] = (ptr[i] - m) / s;
                    }
                }
            }
        }
        
        // NHWC as input
        if (!_cfg.nchw) {
            cv::Mat blob_nhwc;
            std::vector<int> order = {0, 2, 3, 1};  
            // NCHW -> NHWC
            cv::transposeND(blob, order, blob_nhwc);

            // [batch, _cfg.input_h, _cfg.input_w, 3]
            return blob_nhwc;
        }

        return blob;
    }

    std::vector<cv::Mat> YoloTask::inference(const cv::Mat& blob) {
        return (*_rt).inference(blob);
    }

    std::vector<yolo::YoloResult> YoloTask::postprocess(const std::vector<cv::Mat>& raw_outputs, int batch_size) {
        std::vector<yolo::YoloResult> results(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            /* MUST override in child class(extract structured data to fill YoloResult) */
            postprocess_one(raw_outputs, 
                i, _orig_sizes[i], 
                _letterbox_infos[i], results[i]);
        }
        return results;
    }

    std::vector<yolo::YoloResult> YoloTask::run(const std::vector<cv::Mat>& images) {
        /* step1. preprocess */
        auto t1 = std::chrono::system_clock::now();
        auto batch_blob = preprocess(images);
        /* step2. inference */
        auto t2 = std::chrono::system_clock::now();
        auto raw_outputs = inference(batch_blob);
        /* step3. postprocess */
        auto t3 = std::chrono::system_clock::now();
        auto results = postprocess(raw_outputs, images.size());
        auto t4 = std::chrono::system_clock::now();

        assert(images.size() == results.size());
        /* update properties for YoloResult */
        for(int i = 0; i < results.size(); ++i) {
            auto& r = results[i];

            /* note, it's batch cost time here since we do not know the time for single inference. */
            r.speed.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / 1000.0);  // preprocess time(ms)
            r.speed.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() / 1000.0);  // inference time(ms)
            r.speed.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count() / 1000.0);  // postprocess time(ms)

            r.id             = i;                      // batch id
            r.names          = _cfg.names;
            r.batch_size     = _cfg.batch_size;
            r.task           = _cfg.task;
            r.version        = _cfg.version;
            r.target_rt      = _cfg.target_rt;
            r.input_w        = _cfg.input_w;
            r.input_h        = _cfg.input_h;
            r.letterbox_info = _letterbox_infos[i];
            r.input_image    = _input_images[i];
            r.orig_size      = _orig_sizes[i];
            r.orig_image     = images[i];
        }

        /* return results */
        return results;
    }

    std::vector<yolo::YoloResult> YoloTask::operator()(const std::vector<cv::Mat>& images) {
        return run(images);
    }
}