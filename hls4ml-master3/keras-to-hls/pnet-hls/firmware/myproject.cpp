//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "parameters.h"
#include "myproject.h"

#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_batchnorm.h"
#include "nnet_activation.h"
//#include "nnet_pooling.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/a1.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/a3.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/a4.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w6.h"
#include "weights/b6.h"

void myproject(
		  input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1],
		  result_t res[N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    #pragma HLS INTERFACE ap_vld port=data,res 
    #pragma HLS PIPELINE 


    const_size_in   = IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1;
    const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer1_t layer1_out[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    layer1_t conv2d_layer1_out[OUT_HEIGHT_1][OUT_WIDTH_1][N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer1_out complete dim=0
    nnet::conv_2d<input_t, layer1_t, config1>(data, conv2d_layer1_out, w1, b1);
    layer1_t logits1[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
    nnet::flatten<layer1_t, OUT_HEIGHT_1, OUT_WIDTH_1, N_FILT_1>(conv2d_layer1_out, logits1);
    nnet::prelu<layer1_t, layer1_t, PReLU_config1>(logits1, a1, layer1_out);

    layer1_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    layer1_t pool2d_layer2_in[IN_HEIGHT_2][IN_WIDTH_2][N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=pool2d_layer2_in complete dim=0
    nnet::unflatten<layer1_t, IN_HEIGHT_2, IN_WIDTH_2, N_FILT_2>(layer1_out, pool2d_layer2_in);
    layer1_t pool2d_layer2_out[OUT_HEIGHT_2][OUT_WIDTH_2][N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=pool2d_layer2_out complete dim=0
    nnet::pooling2d<layer1_t, config2>(pool2d_layer2_in, pool2d_layer2_out);
    nnet::flatten<layer1_t, OUT_HEIGHT_2, OUT_WIDTH_2, N_FILT_2>(pool2d_layer2_out, layer2_out);

    layer3_t layer3_out[OUT_HEIGHT_3*OUT_WIDTH_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    layer3_t conv2d_layer3_out[OUT_HEIGHT_3][OUT_WIDTH_3][N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer3_out complete dim=0
    nnet::conv_2d<layer2_t, layer3_t, config3>(layer2_out, conv2d_layer3_out, w3, b3);
    layer3_t logits3[OUT_HEIGHT_3*OUT_WIDTH_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=logits3 complete dim=0
    nnet::flatten<layer3_t, OUT_HEIGHT_3, OUT_WIDTH_3, N_FILT_3>(conv2d_layer3_out, logits3);
    nnet::prelu<layer3_t, layer3_t, PReLU_config3>(logits3, a3, layer3_out);

    layer4_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    layer3_t conv2d_layer4_in[IN_HEIGHT_4][IN_WIDTH_4][N_CHAN_4];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer4_in complete dim=0
    nnet::unflatten<layer3_t, IN_HEIGHT_4, IN_WIDTH_4, N_CHAN_4>(layer3_out, conv2d_layer4_in);
    layer4_t conv2d_layer4_out[OUT_HEIGHT_4][OUT_WIDTH_4][N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer4_out complete dim=0
    nnet::conv_2d<layer3_t, layer4_t, config4>(conv2d_layer4_in, conv2d_layer4_out, w4, b4);
    layer4_t logits4[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=logits4 complete dim=0
    nnet::flatten<layer4_t, OUT_HEIGHT_4, OUT_WIDTH_4, N_FILT_4>(conv2d_layer4_out, logits4);
    nnet::prelu<layer4_t, layer4_t, PReLU_config4>(logits4, a4, layer4_out);

    layer5_t layer5_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    layer4_t conv2d_layer5_in[IN_HEIGHT_5][IN_WIDTH_5][N_CHAN_5];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer5_in complete dim=0
    nnet::unflatten<layer4_t, IN_HEIGHT_5, IN_WIDTH_5, N_CHAN_5>(layer4_out, conv2d_layer5_in);
    layer5_t conv2d_layer5_out[OUT_HEIGHT_5][OUT_WIDTH_5][N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer5_out complete dim=0
    nnet::conv_2d<layer4_t, layer5_t, config5>(conv2d_layer5_in, conv2d_layer5_out, w5, b5);
    layer5_t logits5[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=logits5 complete dim=0
    nnet::flatten<layer5_t, OUT_HEIGHT_5, OUT_WIDTH_5, N_FILT_5>(conv2d_layer5_out, logits5);
    nnet::linear<layer5_t, layer5_t, linear_config5>(logits5, layer5_out);

    layer5_t conv2d_layer6_in[IN_HEIGHT_6][IN_WIDTH_6][N_CHAN_6];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer6_in complete dim=0
    nnet::unflatten<layer5_t, IN_HEIGHT_6, IN_WIDTH_6, N_CHAN_6>(layer5_out, conv2d_layer6_in);
    layer6_t conv2d_layer6_out[OUT_HEIGHT_6][OUT_WIDTH_6][N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer6_out complete dim=0
    nnet::conv_2d<layer5_t, layer6_t, config6>(conv2d_layer6_in, conv2d_layer6_out, w6, b6);
    layer6_t logits6[OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=logits6 complete dim=0
    nnet::flatten<layer6_t, OUT_HEIGHT_6, OUT_WIDTH_6, N_FILT_6>(conv2d_layer6_out, logits6);
    nnet::softmax<layer6_t, layer6_t, Softmax_config6>(logits6, layer6_out);


}
