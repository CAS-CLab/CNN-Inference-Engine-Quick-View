# CNN-Inference-Engine-Quick-View
A quick view of high-performance convolution neural networks (CNNs) inference engines.

### FLOAT32-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Intel-Caffe](https://github.com/intel/caffe) | CPU (**Intel** optimized) | Caffe | Y | [Link](https://github.com/intel/caffe/wiki/INTEL%C2%AE-OPTIMIZED-CAFFE-PERFORMANCE-AND-CONVERGENCE)
| [NCNN](https://github.com/Tencent/ncnn) | CPU (**ARM** optimized) | Caffe / pytorch / mxnet / onnx | Y | [Link](https://github.com/Tencent/ncnn/tree/master/benchmark)
| [FeatherCNN](https://github.com/Tencent/FeatherCNN) | CPU (**ARM** optimized) | Caffe | N | [Link](https://github.com/Tencent/FeatherCNN/wiki/Benchmarks) / [unofficial Link](https://www.zhihu.com/question/276372408)
| [Tengine](https://github.com/OAID/Tengine) | CPU (**ARM A72** optimized) | Caffe / mxnet  | Y | [Link](https://github.com/OAID/Tengine/blob/master/doc/benchmark.md)
| [Tensorflowlite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite) | CPU (**Android** optimized) | Caffe2 / Tensorflow / onnx  | Y | [Link](https://www.tensorflow.org/mobile/tflite/performance)
| [TensorRT](https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/) | GPU (**Volta** optimized) | Caffe / Tensorflow / onnx  | Y | [Link](http://on-demand.gputechconf.com/gtc-eu/2017/presentation/23425-han-vanholder-efficient-inference-with-tensorrt.pdf)
| [TVM](https://github.com/dmlc/tvm) | CPU (**ARM** optimized) / Mali GPU / FPGA | onnx  | Y | -
| [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | CPU (**Qualcomm** optimized) / GPU / DSP | Caffe / Caffe2 / Tensorflow/ onnx  | Y | [Link](https://developer.qualcomm.com/docs/snpe/benchmarking.html#benchmarking_overview)

### INT8-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Intel-Caffe](https://github.com/intel/caffe) | CPU (**Intel Skylake**) | Caffe | Y | [Link](https://github.com/intel/caffe/wiki/INTEL%C2%AE-OPTIMIZED-CAFFE-PERFORMANCE-AND-CONVERGENCE)
| [NCNN](https://github.com/Tencent/ncnn) | CPU (**ARM**) | Caffe / pytorch / mxnet / onnx | Y | [Link](https://github.com/Tencent/ncnn/tree/master/benchmark)
| [Tensorflowlite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite) | CPU (**Android**) | Caffe2 / Tensorflow / onnx  | Y | [Link](https://www.tensorflow.org/mobile/tflite/performance)
| [TensorRT](https://devblogs.nvidia.com/tensorrt-3-faster-tensorflow-inference/) | GPU (**Volta**) | Caffe / Tensorflow / onnx  | Y | [Link](http://on-demand.gputechconf.com/gtc-eu/2017/presentation/23425-han-vanholder-efficient-inference-with-tensorrt.pdf)
| [Gemmlowp](https://github.com/google/gemmlowp) | CPU (ARM / x86) | GEMM Library  | - | -
| [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | DSP (Quantized DLC) | Caffe / Caffe2 / Tensorflow/ onnx  | Y | [Link](https://developer.qualcomm.com/docs/snpe/benchmarking.html#benchmarking_overview)

### TERNARY-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [Gemmbitserial](https://github.com/maltanar/gemmbitserial) | CPU (ARM / x86) | GEMM Library | - | [Link](http://www.idi.ntnu.no/%7Eyamanu/2017-cases-wip-quantizedmm-preprint.pdf)

### BINARY-Support
| Framework | Main Platform | Model Compatibility | Detection-Support | Speed Benchmarks
| :----------- | :------: | :------------: | :------------: | :------------:
| [BMXNET](https://github.com/hpi-xnor/BMXNet) | CPU (ARM / x86) / GPU | mxnet | Y | [Link](https://arxiv.org/abs/1705.09864)
| [Espresso](https://github.com/fpeder/espresso) | GPU | - | N | [Link](https://arxiv.org/abs/1705.09864)
| [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ) | FPGA (Xilinx PYNQ) | - | N | [Link](https://openreview.net/forum?id=Sk6fD5yCb)

