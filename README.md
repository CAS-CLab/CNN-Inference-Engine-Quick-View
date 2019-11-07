# CNN-Inference-Engine-Quick-View
A quick view of high-performance convolution neural networks (CNNs) inference engines on mobile devices.

### Runtime-speed Comparisons
* [Mobile-AI-Benchmarks](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/258576406)
* [AI-Benchmarks](http://ai-benchmark.com/ranking_detailed.html)

### FLOAT32-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Intel-Caffe](https://github.com/intel/caffe) | CPU (**Intel** optimized) | Caffe | Y | [Link](https://github.com/intel/caffe/wiki/INTEL%C2%AE-OPTIMIZED-CAFFE-PERFORMANCE-AND-CONVERGENCE)
| [NCNN](https://github.com/Tencent/ncnn) | CPU (**ARM** optimized) | Caffe / pytorch / mxnet / onnx | Y | [Link](https://github.com/Tencent/ncnn/tree/master/benchmark) / [unofficial Link](https://github.com/BUG1989/ncnn-benchmark)
| [MNN](https://github.com/alibaba/MNN) | CPU (**ARM** optimized) / Mali GPU | Caffe / Tensorflow / onnx | Y | [Link](https://github.com/alibaba/MNN/blob/master/benchmark/result/2019-6-17.md) 
| [FeatherCNN](https://github.com/Tencent/FeatherCNN) | CPU (**ARM** optimized) | Caffe | N | [Link](https://github.com/Tencent/FeatherCNN/wiki/Benchmarks) / [unofficial Link](https://www.zhihu.com/question/276372408)
| [Tngine](https://github.com/OAID/Tengine) | CPU (**ARM A72** optimized) | Caffe / mxnet  | Y | [Link](https://github.com/OAID/Tengine/blob/master/doc/benchmark.md)
| [Tensorflowlite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite) | CPU (**Android** optimized) | Caffe2 / Tensorflow / onnx  | Y | [Link](https://www.tensorflow.org/mobile/tflite/performance)
| [TensorRT](https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/) | GPU (**Volta** optimized) | Caffe / Tensorflow / onnx  | Y | [Link](http://on-demand.gputechconf.com/gtc-eu/2017/presentation/23425-han-vanholder-efficient-inference-with-tensorrt.pdf)
| [TVM](https://github.com/dmlc/tvm) | CPU (**ARM** optimized) / Mali GPU / FPGA | onnx  | Y | -
| [SNPE](https://developer.qualcomm.com/docs/snpe/index.html) | CPU (**Qualcomm** optimized) / GPU / DSP | Caffe / Caffe2 / Tensorflow/ onnx  | Y | [Link](https://developer.qualcomm.com/docs/snpe/benchmarking.html#benchmarking_overview)
| [MACE](https://github.com/XiaoMi/mace) | CPU (**ARM** optimized) / Mali GPU / DSP | Caffe / Tensorflow / onnx  | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/151820514)
| [Easy-MACE](https://github.com/conansherry/easy_mace) | CPU (**ARM** optimized) / CPU (**x86** optimized) | Caffe / Tensorflow / onnx  | Y | -
| [In-Prestissimo](https://github.com/in66-dev/In-Prestissimo) | CPU (**ARM** optimized) | Caffe  | N | [Link](https://github.com/in66-dev/In-Prestissimo)
| [Paddle-Mobile](https://github.com/PaddlePaddle/paddle-mobile) | CPU (**ARM** optimized) / Mali GPU / FPGA | Paddle / Caffe / onnx | Y| -
| [Anakin](https://github.com/PaddlePaddle/Anakin) | CPU (**ARM** optimized) / GPU / CPU (**x86** optimized) | Caffe / Fluid | Y| [Link](https://github.com/PaddlePaddle/Anakin/blob/developing/benchmark/README.md)
| [Pocket-Tensor](https://github.com/GValiente/pocket-tensor) | CPU (**ARM**/**x86** optimized) | Keras | N | [Link](https://github.com/GValiente/pocket-tensor)
| [ZQCNN](https://github.com/zuoqing1988/ZQCNN-v0.0) | CPU |  Caffe / mxnet | Y| [Link](https://github.com/zuoqing1988/ZQCNN-v0.0)
| [ARM-NEON-to-x86-SSE](https://github.com/intel/ARM_NEON_2_x86_SSE) | CPU (**Intel** optimized) | Intrinsics-Level | - | -
| [Simd](https://github.com/ermig1979/Simd) | CPU (all platform optimized) | Intrinsics-Level | - | -
| [clDNN](https://github.com/intel/clDNN) |  Intel® Processor Graphics / Iris™ Pro Graphics |  Caffe / Tennsorflow / mxnet / onnx | Y | [Link](https://software.intel.com/en-us/articles/accelerate-deep-learning-inference-with-integrated-intel-processor-graphics-rev-2-0)


### FIX16-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [ARM32-SGEMM-LIB](https://github.com/JunLee85/ARM32-SGEMM-LIB) | CPU (**ARM** optimized) | GEMM Library  | N | [Link](https://github.com/JunLee85/ARM32-SGEMM-LIB/wiki)
| [Yolov2-Xilinx-PYNQ](https://github.com/dhm2013724/yolov2_xilinx_fpga) | FPGA (Xilinx PYNQ) | Yolov2-only | Y | [Link](https://github.com/dhm2013724/yolov2_xilinx_fpga) 

### INT8-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Intel-Caffe](https://github.com/intel/caffe) | CPU (**Intel Skylake**) | Caffe | Y | [Link](https://github.com/intel/caffe/wiki/INTEL%C2%AE-OPTIMIZED-CAFFE-PERFORMANCE-AND-CONVERGENCE)
| [NCNN](https://github.com/Tencent/ncnn) | CPU (**ARM**) | Caffe / pytorch / mxnet / onnx | Y | [Link](https://github.com/Tencent/ncnn/tree/master/benchmark)
| [Tensorflowlite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite) | CPU (**Android**) | Caffe2 / Tensorflow / onnx  | Y | [Link](https://www.tensorflow.org/mobile/tflite/performance)
| [TensorRT](https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/) | GPU (**Volta**) | Caffe / Tensorflow / onnx  | Y | [Link](http://on-demand.gputechconf.com/gtc-eu/2017/presentation/23425-han-vanholder-efficient-inference-with-tensorrt.pdf)
| [Gemmlowp](https://github.com/google/gemmlowp) | CPU (ARM / x86) | GEMM Library  | - | -
| [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | DSP (Quantized DLC) | Caffe / Caffe2 / Tensorflow/ onnx  | Y | [Link](https://developer.qualcomm.com/docs/snpe/benchmarking.html#benchmarking_overview)
| [MACE](https://github.com/XiaoMi/mace) | CPU (**ARM** optimized) / Mali GPU / DSP | Caffe / Tensorflow / onnx  | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/87212589)
| [In-Prestissimo](https://github.com/in66-dev/In-Prestissimo) | CPU (**ARM** optimized) | Caffe  | N | [Link](https://github.com/in66-dev/In-Prestissimo)
| [Paddle-Mobile](https://github.com/PaddlePaddle/paddle-mobile) | CPU (**ARM** optimized) / Mali GPU / FPGA | Paddle / Caffe / onnx | Y| -
| [Anakin](https://github.com/PaddlePaddle/Anakin) | CPU (**ARM** optimized) / GPU / CPU (**x86** optimized) | Caffe / Fluid | Y| [Link](https://github.com/PaddlePaddle/Anakin/blob/developing/benchmark/README.md)

### TERNARY-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Gemmbitserial](https://github.com/maltanar/gemmbitserial) | CPU (ARM / x86) | GEMM Library | - | [Link](http://www.idi.ntnu.no/%7Eyamanu/2017-cases-wip-quantizedmm-preprint.pdf)

### BINARY-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [BMXNET](https://github.com/hpi-xnor/BMXNet) | CPU (ARM / x86) / GPU | mxnet | Y | [Link](https://arxiv.org/abs/1705.09864)
| [DABNN](https://github.com/JDAI-CV/dabnn) | CPU (ARM) | Caffe / Tensorflow / onnx | N | [Link](https://github.com/JDAI-CV/dabnn/blob/master/images/comparison_en.png)
| [Espresso](https://github.com/fpeder/espresso) | GPU | - | N | [Link](https://arxiv.org/abs/1705.09864)
| [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ) | FPGA (Xilinx PYNQ) | - | N | [Link](https://openreview.net/forum?id=Sk6fD5yCb)
| [FINN](https://github.com/Xilinx/FINN) | FPGA (Xilinx) | - | N | [Link](https://arxiv.org/abs/1612.07119)

### MobileNet-v1 Speed Benchmarks on RK3399
Rockchip RK3399 (Cortex-A72 1.8GHz x 2 + Cortex-A53 1.5GHz x 4):

Framework (ms) | 1 Thread  | 2 Threads | 3 Threads | 4 Threads
------------ | ------------- | ------------ | ------------- | ----------
Caffe+OpenBLAS`*` | 250.57 | 204.40 | 248.65 | 230.20
FeatherCNN | 205.76 | 135.17 | **183.34** | **194.67**
NCNN`**` | 150.95 | 90.79 | 232.31 | 231.64
NCNN-Opt | 122.22 | 67.47 | - | -
Tengine | 122.10 | 65.42 | - | -
Tengine-Opt | **115.29** | **63.94** | - | -

`*`: optimized for Cortex-A53 instead of Cortex-A72

`**`: powersave=0

For 1 Thread, we set task on a single A72, and A72 x 2 for 2 Threads.  

### ResNet-18 Speed Benchmarks on RK3399

Framework (ms) | 1 Thread  | 2 Threads | 8 Threads
------------ | ------------- | ------------ | ----------
NCNN`*` | 340.33 | 211.78 | -
NCNN-Opt | **332.20** | **206.62**  | **196.97**
Tengine | 402.57 | 226.02 |  -

`*`: [Conv-BN-Scale-fused](https://github.com/HolmesShuan/Caffe-Computation-Graph-Optimization)

