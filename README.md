# CNN-Inference-Engine-Quick-View
A quick view of high-performance convolution neural networks (CNNs) inference engines on mobile devices.

### Runtime-speed Comparisons
* [Mobile-AI-Benchmarks](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/526762368)
* [AI-Benchmarks](http://ai-benchmark.com/ranking_detailed.html)

### FLOAT32-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Bolt](https://github.com/huawei-noah/bolt) | CPU (**ARM** optimized) / x86 / Mali GPU | Caffe / Tensorflow / PyTorch / onnx | Y | [Link](https://github.com/huawei-noah/bolt/blob/master/docs/BENCHMARK.md) 
| [TNN](https://github.com/Tencent/TNN) | CPU (**ARM** optimized) / Mali Adreno Apple GPU | Caffe / Tensorflow / PyTorch | Y | [Link](https://github.com/Tencent/TNN/blob/master/doc/en/development/profiling_en.md) 
| [Paddle-Light](https://github.com/PaddlePaddle/Paddle-Lite) | CPU (**ARM** optimized) / Mali GPU / FPGA / **NPU** | Paddle / Caffe / onnx | Y| [Link](https://paddlepaddle.github.io/Paddle-Lite/develop/benchmark/)
| [MNN](https://github.com/alibaba/MNN) | CPU (**ARM** optimized) / Mali GPU | Caffe / Tensorflow / onnx | Y | [Link](https://github.com/alibaba/MNN/blob/master/benchmark/result/2020-3-22.md) 
| [NCNN](https://github.com/Tencent/ncnn) | CPU (**ARM** optimized) / Mali GPU | Caffe / pytorch / mxnet / onnx | Y | [3rd party Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/402963978) / [Official Link](https://github.com/Tencent/ncnn/tree/master/benchmark)
| [MACE](https://github.com/XiaoMi/mace) | CPU (**ARM** optimized) / Mali GPU / DSP | Caffe / Tensorflow / onnx  | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/151820514)
| [ONNX-Runtime](https://github.com/microsoft/onnxruntime) | CPU / Nvidia GPU | onnx  | Y | -
| [HiAI](https://developer.huawei.com/consumer/cn/hiai) | Kirin CPU / NPU | Caffe / Tensorflow | Y | -
| [NNIE](https://github.com/RaySue/NNIE-lite) | NPU | Caffe | Y | [1TOPs](http://www.hisilicon.com/-/media/Hisilicon/pdf/Surveillance_mobilecam/Hi3516DV300.pdf)
| [Intel-Caffe](https://github.com/intel/caffe) | CPU (**Intel** optimized) | Caffe | Y | [Link](https://github.com/intel/caffe/wiki/INTEL%C2%AE-OPTIMIZED-CAFFE-PERFORMANCE-AND-CONVERGENCE)
| [FeatherCNN](https://github.com/Tencent/FeatherCNN) | CPU (**ARM** optimized) | Caffe | N | [Link](https://github.com/Tencent/FeatherCNN/wiki/Benchmarks) / [unofficial Link](https://www.zhihu.com/question/276372408)
| [TEngine](https://github.com/OAID/Tengine) | CPU (**ARM A72** optimized) | Caffe / mxnet  | Y | [Link](https://github.com/OAID/Tengine/blob/master/doc/benchmark.md)
| [Tensorflowlite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite) | CPU (**Android** optimized) | Caffe2 / Tensorflow / onnx  | Y | [Link](https://www.tensorflow.org/mobile/tflite/performance)
| [TensorRT](https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/) | GPU (**Volta** optimized) | Caffe / Tensorflow / onnx  | Y | [Link](http://on-demand.gputechconf.com/gtc-eu/2017/presentation/23425-han-vanholder-efficient-inference-with-tensorrt.pdf)
| [TVM](https://github.com/dmlc/tvm) | CPU (**ARM** optimized) / Mali GPU / FPGA | onnx  | Y | -
| [SNPE](https://developer.qualcomm.com/docs/snpe/index.html) | CPU (**Qualcomm** optimized) / GPU / DSP | Caffe / Caffe2 / Tensorflow/ onnx  | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/402963978)
| [Pocket-Tensor](https://github.com/GValiente/pocket-tensor) | CPU (**ARM**/**x86** optimized) | Keras | N | [Link](https://github.com/GValiente/pocket-tensor)
| [ZQCNN](https://github.com/zuoqing1988/ZQCNN-v0.0) | CPU |  Caffe / mxnet | Y| [Link](https://github.com/zuoqing1988/ZQCNN-v0.0)
| [ARM-NEON-to-x86-SSE](https://github.com/intel/ARM_NEON_2_x86_SSE) | CPU (**Intel** optimized) | Intrinsics-Level | - | -
| [Simd](https://github.com/ermig1979/Simd) | CPU (all platform optimized) | Intrinsics-Level | - | -
| [clDNN](https://github.com/intel/clDNN) |  Intel® Processor Graphics / Iris™ Pro Graphics |  Caffe / Tennsorflow / mxnet / onnx | Y | [Link](https://software.intel.com/en-us/articles/accelerate-deep-learning-inference-with-integrated-intel-processor-graphics-rev-2-0)

### FIX16-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Bolt](https://github.com/huawei-noah/bolt) | CPU (**ARM** optimized) / x86 / Mali GPU | Caffe / Tensorflow / PyTorch | Y | [Link](https://github.com/huawei-noah/bolt/blob/master/docs/BENCHMARK.md) 
| [ARM32-SGEMM-LIB](https://github.com/JunLee85/ARM32-SGEMM-LIB) | CPU (**ARM** optimized) | GEMM Library  | N | [Link](https://github.com/JunLee85/ARM32-SGEMM-LIB/wiki)
| [TNN](https://github.com/Tencent/TNN) | CPU (**ARM** optimized) / Mali Adreno Apple GPU | Caffe / Tensorflow / PyTorch | Y | [Link](https://github.com/Tencent/TNN/blob/master/doc/en/development/profiling_en.md) 
| [Yolov2-Xilinx-PYNQ](https://github.com/dhm2013724/yolov2_xilinx_fpga) | FPGA (Xilinx PYNQ) | Yolov2-only | Y | [Link](https://github.com/dhm2013724/yolov2_xilinx_fpga) 

### INT8-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Bolt](https://github.com/huawei-noah/bolt) | CPU (**ARM** optimized) / x86 / Mali GPU | Caffe / Tensorflow / PyTorch | Y | [Link](https://github.com/huawei-noah/bolt/blob/master/docs/BENCHMARK.md) 
| [Intel-Caffe](https://github.com/intel/caffe) | CPU (**Intel Skylake**) | Caffe | Y | [Link](https://github.com/intel/caffe/wiki/INTEL%C2%AE-OPTIMIZED-CAFFE-PERFORMANCE-AND-CONVERGENCE)
| [TNN](https://github.com/Tencent/TNN) | CPU (**ARM** optimized) / Mali Adreno Apple GPU | Caffe / Tensorflow / PyTorch | Y | [Link](https://github.com/Tencent/TNN/blob/master/doc/en/development/profiling_en.md) 
| [NCNN](https://github.com/Tencent/ncnn) | CPU (**ARM**) | Caffe / pytorch / mxnet / onnx | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/402963978)
| [Paddle-Light](https://github.com/PaddlePaddle/Paddle-Lite) | CPU (**ARM** optimized) / Mali GPU / FPGA | Paddle / Caffe / onnx | Y| [Link](https://paddlepaddle.github.io/Paddle-Lite/develop/benchmark/)
| [MNN](https://github.com/alibaba/MNN) | CPU (**ARM** optimized) / Mali GPU | Caffe / Tensorflow / onnx | Y | [Link](https://github.com/alibaba/MNN/blob/master/benchmark/result/2020-3-22.md) 
| [Tensorflowlite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite) | CPU (**Android**) | Caffe2 / Tensorflow / onnx  | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/402963978)
| [TensorRT](https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/) | GPU (**Volta**) | Caffe / Tensorflow / onnx  | Y | [Link](http://on-demand.gputechconf.com/gtc-eu/2017/presentation/23425-han-vanholder-efficient-inference-with-tensorrt.pdf)
| [Gemmlowp](https://github.com/google/gemmlowp) | CPU (ARM / x86) | GEMM Library  | - | -
| [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | DSP (Quantized DLC) | Caffe / Caffe2 / Tensorflow/ onnx  | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/402963978)
| [MACE](https://github.com/XiaoMi/mace) | CPU (**ARM** optimized) / Mali GPU / DSP | Caffe / Tensorflow / onnx  | Y | [Link](https://gitlab.com/llhe/mobile-ai-bench/-/jobs/402963978)
| [TF2](https://github.com/TF2-Engine/TF2) | FPGA | Caffe / PyTorch / Tensorflow | Y| [Link](https://github.com/TF2-Engine/TF2#runtime-engine)

### TERNARY-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Gemmbitserial](https://github.com/maltanar/gemmbitserial) | CPU (ARM / x86) | GEMM Library | - | [Link](http://www.idi.ntnu.no/%7Eyamanu/2017-cases-wip-quantizedmm-preprint.pdf)

### BINARY-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Bolt](https://github.com/huawei-noah/bolt) | CPU (**ARM** optimized) / x86 / Mali GPU | Caffe / Tensorflow / PyTorch | Y | [Link](https://github.com/huawei-noah/bolt/blob/master/docs/BENCHMARK.md) 
| [BMXNET](https://github.com/hpi-xnor/BMXNet) | CPU (ARM / x86) / GPU | mxnet | Y | [Link](https://arxiv.org/abs/1705.09864)
| [DABNN](https://github.com/JDAI-CV/dabnn) | CPU (ARM) | Caffe / Tensorflow / onnx | N | [Link](https://github.com/JDAI-CV/dabnn/blob/master/images/comparison_en.png)
| [Espresso](https://github.com/fpeder/espresso) | GPU | - | N | [Link](https://arxiv.org/abs/1705.09864)
| [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ) | FPGA (Xilinx PYNQ) | - | N | [Link](https://openreview.net/forum?id=Sk6fD5yCb)
| [FINN](https://github.com/Xilinx/FINN) | FPGA (Xilinx) | - | N | [Link](https://arxiv.org/abs/1612.07119)


### NLP-Support
| Framework | Main Platform | Model Compatibility | Speed Benchmarks
| :----------- | :------: | :------------: | :------------:
| [TurboTransformers](https://github.com/Tencent/TurboTransformers) | CPU / Nvidia GPU | PyTorch | [Link](https://github.com/Tencent/TurboTransformers#performance)

`*`: [Conv-BN-Scale-fused](https://github.com/HolmesShuan/Caffe-Computation-Graph-Optimization)

