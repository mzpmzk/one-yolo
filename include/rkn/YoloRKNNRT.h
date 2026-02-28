#pragma once

#include <rknn_api.h>
#include "YoloRuntime.h"

namespace yolo {
    /**
     * @brief
     * Yolo runtime based on rknn library.
    */
    class YoloRKNNRT: public YoloRuntime
    {
    private:
        rknn_context __ctx = 0;
        rknn_input_output_num __io_num;
        rknn_tensor_attr __input_attr;
        std::vector<rknn_tensor_attr> __output_attrs;

        bool queryIO();
    public:
        YoloRKNNRT(const std::string& model_path);
        ~YoloRKNNRT();
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) override;
    };
}