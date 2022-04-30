#include <torch/extension.h>
#include <ATen/DeviceGuard.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int sparse_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor output,
                             at::Tensor columns, at::Tensor ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH, int group,
                             int im2col_step);

int sparse_conv_backward_input_cuda(at::Tensor input,
                                    at::Tensor gradOutput, at::Tensor gradInput,
                                    at::Tensor weight,
                                    at::Tensor columns, int kW, int kH, int dW,
                                    int dH, int padW, int padH, int dilationW,
                                    int dilationH, int group,
                                    int im2col_step);

int sparse_conv_backward_parameters_cuda(
    at::Tensor input, at::Tensor gradOutput,
    at::Tensor gradWeight,
    at::Tensor columns, at::Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    float scale, int im2col_step, float s);

#endif

int sparse_conv_forward(at::Tensor input, at::Tensor weight,
                             at::Tensor output,
                             at::Tensor columns, at::Tensor ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH, int group,
                             int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return sparse_conv_forward_cuda(input, weight, output, columns,
        ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, group,
        im2col_step);
#else
    AT_ERROR("sparse conv is not compiled with GPU support");
#endif
  }
  AT_ERROR("sparse conv is not implemented on CPU");
}

int sparse_conv_backward_input(at::Tensor input,
                                    at::Tensor gradOutput, at::Tensor gradInput,
                                    at::Tensor weight,
                                    at::Tensor columns, int kW, int kH, int dW,
                                    int dH, int padW, int padH, int dilationW,
                                    int dilationH, int group,
                                    int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return sparse_conv_backward_input_cuda(input, gradOutput,
        gradInput, weight, columns, kW, kH, dW, dH, padW, padH,
        dilationW, dilationH, group, im2col_step);
#else
    AT_ERROR("sparse conv is not compiled with GPU support");
#endif
  }
  AT_ERROR("sparse conv is not implemented on CPU");
}

int sparse_conv_backward_parameters(
    at::Tensor input, at::Tensor gradOutput,
    at::Tensor gradWeight,
    at::Tensor columns, at::Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    float scale, int im2col_step, float s) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return sparse_conv_backward_parameters_cuda(input, gradOutput,
        gradWeight, columns, ones, kW, kH, dW, dH, padW, padH, dilationW,
        dilationH, group, scale, im2col_step, s);
#else
    AT_ERROR("sparse conv is not compiled with GPU support");
#endif
  }
  AT_ERROR("sparse conv is not implemented on CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_conv_forward", &sparse_conv_forward,
        "sparse forward");
  m.def("sparse_conv_backward_input", &sparse_conv_backward_input,
        "sparse_conv_backward_input");
  m.def("sparse_conv_backward_parameters",
        &sparse_conv_backward_parameters,
        "sparse_conv_backward_parameters");
}