
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace yolo {
    /**
     * @brief
     * base class for all Yolo runtimes.
     * the main duty is running Yolo model with input tensor and get raw output tensors.
     * 
     * two stuffs to care:
     * 1. which inference backend you want to use? opencv::dnn, onnxruntime, or tenssorrt...
     * 2. which hardware platform you want to run? nvidia, ascend, or rockchip...
    */
    class YoloRuntime {
    private:
        std::string __rt_name = "default_rt";
    public:
        YoloRuntime(const std::string& rt_name);
        ~YoloRuntime();
        /**
         * @brief
         * inference based on different Yolo runtimes.
         * 
         * @param blob a 4D matrix to be sent to Yolo network.
         * @return raw output matrixs from Yolo network, support multi-heads.
         * 
         * @note
         * we use `cv::Mat` as the data structure to hold raw input & raw output for the inference, which acts like `Tensor` in `PyTorch` or other deep learning libraries.
         * when using `cv::Mat` as a generic N-dimensional tensor (e.g., for deep learning inference or multi-dimensional array computation),
         * you should discard OpenCV's traditional `image semantics` and adopt `array semantics` instead.
        */
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) = 0;

        /**
         * @brief
         * get description for the specific Yolo runtime.
         * 
         * @return description for the specific Yolo runtime, return `__rt_name` by default.
        */
        virtual std::string to_string();
    };
}