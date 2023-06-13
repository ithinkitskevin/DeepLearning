from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] += lam * self.params[n]


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        batch_size = input_size[0]
        output_height = (((input_size[1] - self.kernel_size) + (2 * self.padding)) // self.stride) + 1
        output_width = (((input_size[2] - self.kernel_size) + (2 * self.padding)) // self.stride) + 1
        filter_number = self.number_filters
#         print(batch_size, output_height, output_width, filter_number)
        
        output_shape = [int(batch_size), int(output_height), int(output_width), int(filter_number)]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.         #
        # Store the results in the variable "output" provided above.                #
        #############################################################################   
#         print(output_shape)
        output = np.zeros(output_shape)
        padded_image = np.pad(img, ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        for h in range(output_height):
            height_start = self.stride * h
            height_end = self.stride * h + self.kernel_size
            for w in range(output_width):
                width_start = self.stride * w
                width_end = self.stride * w + self.kernel_size

                kernel = padded_image[:,height_start:height_end, width_start:width_end,:, np.newaxis]
#                 print(kernel)
                weights = self.params[self.w_name][np.newaxis:,:,:]
#                 print(weights)
                
                o = kernel * weights
                activation = np.sum(o, axis=(1,2,3))  # axis=()
                
                output[:,h,w,:] = activation
            
        output += self.params[self.b_name]          
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        _, h_out, w_out, _ = dprev.shape
        
        self.grads[self.w_name] = np.zeros((self.kernel_size, self.kernel_size, self.input_channels,self.number_filters))
        self.grads[self.b_name] = np.sum(dprev,axis=(0,1,2)) 
        
        dimg = np.zeros(img.shape)
        dimg_pad = np.pad(dimg, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), constant_values = (0,0))
        img_pad = np.pad(img, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), constant_values = (0,0))
        
        for h in range(h_out):
            height_start = self.stride * h
            height_end = self.stride * h + self.kernel_size
            for w in range(w_out):
                width_start = self.stride * w
                width_end = self.stride * w + self.kernel_size
                
                kernel = img_pad[:, height_start:height_end, width_start:width_end, :, np.newaxis]
                prev_loc = dprev[:, h:h+1, w:w+1, np.newaxis, :]

                self.grads[self.w_name] += np.sum(kernel * prev_loc,axis=0)

                e = self.params[self.w_name][np.newaxis, :, :, :, :]
                s = np.sum(e * prev_loc, axis=4)

                dimg_pad[:, height_start:height_end, width_start:width_end,:] += s

        dimg = dimg_pad[:, self.padding:self.padding + img.shape[1], self.padding:self.padding + img.shape[1], :]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        f, image_height, image_width, g = img.shape
        output_height = int(((image_height - self.pool_size) // self.stride) + 1)
        output_width = int(((image_width - self.pool_size) // self.stride) + 1)

        output = np.zeros((f, output_height, output_width, g))
        for h in range(output_height):
            height_start = self.stride * h
            height_end = self.stride * h + self.pool_size
            for w in range(output_width):
                width_start = self.stride * w
                width_end = self.stride * w + self.pool_size

                a_slice = img[:, height_start:height_end, width_start:width_end, :]
                output[:,h,w,:] = np.max(a_slice,axis = (1,2))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        for h in range(h_out):
            height_start = self.stride * h
            height_end = self.stride * h + self.pool_size
            for w in range(w_out):
                width_start = self.stride * w
                width_end = self.stride * w + self.pool_size
                
                pool_mult = h_pool * w_pool

                kernel = img[:, height_start:height_end, width_start:width_end, :]
                
                mask = np.zeros_like(kernel)
                f, g = np.indices((kernel.shape[0], kernel.shape[-1]))
                max_kernel = np.argmax(kernel.reshape(dprev.shape[0], pool_mult, dprev.shape[-1]),axis=1)
                mask.reshape(dprev.shape[0], pool_mult, dprev.shape[-1])[f, max_kernel, g] = 1
                
                prev_loc = dprev[:, h:h+1, w:w+1, :]
                
                dimg[:, height_start:height_end, width_start:width_end, :] += mask * prev_loc
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
