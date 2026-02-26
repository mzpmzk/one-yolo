<p style="" align="center">
  <img src="./docs/logo.png" alt="Logo" width="85%">
</p>
<p style="margin:0px;color:gray" align="center">
[🚀🚀🚀支持全部 YOLO 任务 · 支持全部 YOLO 版本 · 支持全部 YOLO 推理后端🚀🚀🚀]
</p>
<p style="margin:0px" align="center">
  <a href='./README.md'>英文README</a> | <a href='https://github.com/sherlockchou86/one-yolo/wiki'>Wiki 文档 </a>
</p>

# one-yolo
一个统一的 C++ YOLO 工具箱，支持 `v5 / v8 / v11 / v26 / ...`，覆盖 `分类 / 检测 / 分割 / 姿态 / OBB（旋转框）` 等任务，提供类似 ultralytics/ultralytics 的 Python 风格易用 API。支持 全部 YOLO 任务、全部 YOLO 版本、全部 YOLO 推理后端 —— 是时候真正做到 All-in-One。
<p style="" align="center">
  <img src="./docs/showcase.gif" alt="Logo" width="85%">
</p>

## ✨ 亮点
1. 支持全部 YOLO 任务：`classification（分类）` / `detection（检测）` / `segmentation（分割）` / `pose（姿态）` / `obb（旋转框）`
2. 支持全部 YOLO 版本：`yolov5（anchor-based）` / `yolov5u（anchor-free）` / `yolov8` / `yolov11` / `yolov26（nms-free）` / `以及未来版本`。同时支持 `n / s / m / l / x` 等子版本
3. 支持全部 YOLO 推理后端（runtime）：`OpenCV::DNN` / `ONNXRuntime` / `TensorRT` / `OpenVINO` / `RKNN` / `CoreML` / `CANN` / `PaddlePaddle` ...
4. API 简洁易用，调用方式类似 `ultralytics/ultralytics` python库
5. 开箱即用：提供模型，设置配置参数，即可开始预测

## 🚀 快速开始

### 基础依赖
1. C++ >= 17
2. GCC >= 7.5
3. OpenCV == 4.13
4. CUDA/ONNXRuntime/TensorRT/OpenVINO/RKNN/... are optional

### 编译源码
1. run `git clone https://github.com/sherlockchou86/one-yolo.git`
2. run `cd one-yolo && mkdir build && cd build`
3. run `cmake .. && make -j8` or click `debug` button to run samples directly if you have opened the project using VS Code

> you must put test data(models&video&images) at the same directory as one-yolo first before runing the samples.

```
在执行cmake命令时可以携带以下选项:
-DBUILD_WITH_ORT=ON   # 启用 ONNXRuntime
-DBUILD_WITH_OVN=ON   # 启用 OpenVINO（Intel 平台）
-DBUILD_WITH_TRT=ON   # 启用 TensorRT（NVIDIA / CUDA平台）
-DBUILD_WITH_RKN=ON   # 启用 RKNN（RockChip 平台）
-DBUILD_WITH_CML=ON   # 启用 CoreML（Apple 平台）
-DBUILD_WITH_PDL=ON   # 启用 PaddlePaddle
-DBUILD_WITH_CAN=ON   # 启用 CANN（华为平台）
-DBUILD_WITH_DEL=ON   # 启用 登临SDK（登临平台）
-DBUILD_WITH_CAB=ON   # 启用 寒武纪SDK（寒武纪平台）
...

如果直接运行 `cmake ..`
默认使用 OpenCV::DNN 作为推理后端，因此one-yolo项目中 OpenCV 为必需依赖。
CUDA 在自行编译 OpenCV 时为可选项。
```

### one-yolo示例

使用 yolov8s 进行车辆检测与跟踪：
```c++
#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. 构建 YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "基于 yolov8s 的车辆检测任务（自定义模型）";
    cfg.version     = YoloVersion::YOLO8;
    cfg.task        = YoloTaskType::DET;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/det_cls/vehicel_v8s-det_c6_20260205.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 384;
    cfg.batch_size  = 1;
    cfg.num_classes = 6;
    cfg.names       = {"person", "car", "bus", "truck", "2wheel", "other"};

    /* 2. 创建Yolo模型 */
    auto model = Yolo(cfg);
    model.info();

    /* 3. 构建 YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.algo = YoloTrackAlgo::SORT;
    t_cfg.iou_thresh = 0.6f;

    /* 4. 创建跟踪器 */
    auto tracker = YoloTracker(t_cfg);
    tracker.info();

    /* 5. 打开视频并循环预测 */
    cv::VideoCapture cap("./vp_data/test_video/rgb.mp4");
    while (cap.isOpened()) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        if (frame.cols > 720) {
            cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        }
        
        // 以batch模式预测（batch size == 1）
        auto results = model(std::vector<cv::Mat>{frame});
        tracker(results[0]);

        // 打印输出结果
        results[0].info();
        results[0].to_json(true);
        results[0].to_csv(true);

        // 显示结果
        if (results[0].show(
            false, 1.0f, DrawParam(),
            true, true) == 27) {
            break;
        }

        /*
         * 你还可以通过下面的方式获取预测的结果:
         * auto boxes        = results[0].boxes();          // get bounding boxes in detection task
         * auto cls_ids      = results[0].cls_ids();        // get class ids in detection task
         * auto confs        = results[0].confs();          // get confidences in detection task
         * auto labels       = results[0].labels();         // get labels in detection task
         * auto track_ids    = results[0].track_ids();      // get track ids in detection task
         * auto track_points = results[0].track_points();   // get track points in detection task
        */
    }
}
```
### 示例效果
使用 yolov8s 进行车辆检测与跟踪的视频效果：

https://github.com/user-attachments/assets/d8b0b711-8922-41f8-8ec7-d1cea1f48afc

### 示例输出
json/csv output result of vechile detection & tracking using yolov8s:
```
json output:
[
    {
        "box": {
            "height": 76,
            "width": 33,
            "x": 368,
            "y": 378
        },
        "cls_id": 4,
        "conf": 0.8655326962471008,
        "label": "2wheel",
        "track_id": 1
    },
    {
        "box": {
            "height": 21,
            "width": 10,
            "x": 647,
            "y": 145
        },
        "cls_id": 4,
        "conf": 0.8104556202888489,
        "label": "2wheel",
        "track_id": 37
    },
    {
        "box": {
            "height": 15,
            "width": 9,
            "x": 676,
            "y": 137
        },
        "cls_id": 4,
        "conf": 0.7772445678710938,
        "label": "2wheel",
        "track_id": 23
    },
    {
        "box": {
            "height": 14,
            "width": 7,
            "x": 710,
            "y": 118
        },
        "cls_id": 4,
        "conf": 0.523908257484436,
        "label": "2wheel",
        "track_id": 41
    },
    {
        "box": {
            "height": 14,
            "width": 12,
            "x": 793,
            "y": 93
        },
        "cls_id": 3,
        "conf": 0.5332302451133728,
        "label": "truck",
        "track_id": 44
    },
    {
        "box": {
            "height": 128,
            "width": 113,
            "x": 494,
            "y": 369
        },
        "cls_id": 1,
        "conf": 0.9514954090118408,
        "label": "car",
        "track_id": 5
    },
    {
        "box": {
            "height": 9,
            "width": 13,
            "x": 721,
            "y": 117
        },
        "cls_id": 1,
        "conf": 0.7941694259643555,
        "label": "car",
        "track_id": 25
    },
    {
        "box": {
            "height": 9,
            "width": 14,
            "x": 753,
            "y": 116
        },
        "cls_id": 1,
        "conf": 0.7911720871925354,
        "label": "car",
        "track_id": 13
    },
    {
        "box": {
            "height": 11,
            "width": 13,
            "x": 770,
            "y": 107
        },
        "cls_id": 1,
        "conf": 0.5813544988632202,
        "label": "car",
        "track_id": 42
    }
]
csv output:
id,cls_id,conf,label,track_id
1,4,0.865533,2wheel,1
2,4,0.810456,2wheel,37
3,4,0.777245,2wheel,23
4,4,0.523908,2wheel,41
5,3,0.53323,truck,44
6,1,0.951495,car,5
7,1,0.794169,car,25
8,1,0.791172,car,13
9,1,0.581354,car,42
```
## 📚 参考资料
wait for update
1. api docs
2. samples
3. to-do
