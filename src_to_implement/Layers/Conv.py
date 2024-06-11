import numpy as np
from Layers import Base
from scipy.signal import correlate, convolve


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape

        if len(convolution_shape) == 2:
            self.is_1d = True
            self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        elif len(convolution_shape) == 3:
            self.is_1d = False
            self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))

        self.bias = np.random.uniform(0, 1, num_kernels)
        self.trainable = True
        self._gradient_weights = None
        self._gradient_bias = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer((self.num_kernels, *self.convolution_shape))
        self.bias = bias_initializer((self.num_kernels,))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batches, channels, *spatial_dims = input_tensor.shape

        if self.is_1d:
            # pad_size = self.convolution_shape[1] // 2
            # input_padded = np.pad(input_tensor, ((0, 0), (0, 0), (pad_size, pad_size)), mode='constant')
            #
            # spatial_dims_h_out = (spatial_dims_h + 2 * pad_size - self.convolution_shape[1]) // self.stride_shape[0] + 1
            # output_tensor_shape = (batches, self.num_kernels, spatial_dims_h_out)
            # output_tensor = np.zeros(output_tensor_shape)

            output_tensor_shape = (batches, self.num_kernels, *spatial_dims)
            output_tensor = np.zeros(output_tensor_shape)

            for batch in range(batches):
                for kernel in range(self.num_kernels):
                    for channel in range(channels):
                        output_tensor[batch, kernel] += correlate(input_tensor[batch, channel], self.weights[kernel, channel], mode='same')

                    output_tensor[batch, kernel] += self.bias[kernel]

            print("self.stride_shapself.stride_shapself.stride_shap", self.stride_shape)
            output_tensor = output_tensor[:, :, ::self.stride_shape[0]]
        else:
            # pad_size = [(self.convolution_shape[i + 1]-1) for i in range(2)]
            # input_padded = np.pad(input_tensor,
            #                       ((0, 0), (0, 0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1])),
            #                       mode='constant')
            #
            # H_out = (spatial_dims[0] + pad_size[0] - self.convolution_shape[1]) // self.stride_shape[0] + 1
            # W_out = (spatial_dims[1] + pad_size[1] - self.convolution_shape[2]) // self.stride_shape[1] + 1
            #
            # # H_out = (input_padded.shape[2]- self.convolution_shape[1]) // self.stride_shape[0] + 1
            # # W_out = (input_padded.shape[3] - self.convolution_shape[2]) // self.stride_shape[1] + 1
            #
            output_tensor_shape = (batches, self.num_kernels, *spatial_dims)
            output_tensor = np.zeros(output_tensor_shape)


            for batch in range(batches):
                for kernel in range(self.num_kernels):
                    for channel in range(channels):
                        output_tensor[batch, kernel] += correlate(input_tensor[batch, channel], self.weights[kernel, channel], mode='same')

                    output_tensor[batch, kernel] += self.bias[kernel]

            output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        self.output_tensor = output_tensor
        return output_tensor

    # def forward(self, input_tensor):
    #     # 1D Order: batches, channels, y (width)
    #     # 2D Order: batches, channel, y (width) , x (height)
    #     batches = input_tensor.shape[0]
    #     self.input_tensor = input_tensor
    #
    #     channels = input_tensor.shape[1]  # Channels of the image (if it's RGB or not)
    #     one_dim_conv = len(input_tensor.shape) == 3  # Flag that is true if it is a 1D Convolution
    #     output_tensor = np.zeros([batches, self.num_kernels, *input_tensor.shape[2:]])
    #
    #     for b in range(batches):  # iterate over each tensor (e.g. image) in the batch
    #         for k in range(self.num_kernels):  # iterate over each kernel
    #             for c in range(
    #                     channels):  # iterate over each channel to sum them up in the end to get 3D convolution (feature map)
    #                 output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='same')
    #
    #             output_tensor[b, k] += self.bias[k]  # add bias to each feature map
    #
    #     if one_dim_conv:
    #         output_tensor = output_tensor[:, :, ::self.stride_shape[0]]
    #     else:
    #         output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
    #
    #     self.output_tensor = output_tensor
    #     return output_tensor

    def backward(self, error_tensor):
        grad_weights = np.zeros_like((self.weights))
        grad_bias = np.sum(error_tensor, axis=(0, 2, 3))
        grad_input = np.zeros_like(self.input_tensor)

        for batch in range(self.input_tensor.shape[0]):
            for channel in range(self.num_kernels):
                grad_weights[channel] += correlate(self.input[batch], error_tensor[batch, channel], mode='reflect')
                grad_input[batch] += convolve(error_tensor[batch, channel], self.weights[channel], mode='reflect')

        return grad_input

    @property
    def optimizer(self):
        return (self.weight_optimizer, self.bias_optimizer)

    @optimizer.setter
    def optimizer(self, value):
        self.weight_optimizer, self.bias_optimizer = value

