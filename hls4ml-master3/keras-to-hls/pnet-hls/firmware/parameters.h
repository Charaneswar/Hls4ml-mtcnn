#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_batchnorm.h"
//#include "nnet_pooling.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<16,6> accum_default_t;
typedef ap_fixed<16,6> weight_default_t;
typedef ap_fixed<16,6> bias_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;
#define IN_HEIGHT_1 320
#define IN_WIDTH_1 240
#define N_CHAN_1 3
#define OUT_HEIGHT_1 318
#define OUT_WIDTH_1 238
#define N_FILT_1 10
#define IN_HEIGHT_2 318
#define IN_WIDTH_2 238
#define OUT_HEIGHT_2 159
#define OUT_WIDTH_2 119
#define POOL_HEIGHT_2 2
#define POOL_WIDTH_2 2
#define N_FILT_2 10
#define N_LAYER_2 189210
#define IN_HEIGHT_3 159
#define IN_WIDTH_3 119
#define N_CHAN_3 10
#define OUT_HEIGHT_3 157
#define OUT_WIDTH_3 117
#define N_FILT_3 16
#define IN_HEIGHT_4 157
#define IN_WIDTH_4 117
#define N_CHAN_4 16
#define OUT_HEIGHT_4 155
#define OUT_WIDTH_4 115
#define N_FILT_4 32
#define IN_HEIGHT_5 155
#define IN_WIDTH_5 115
#define N_CHAN_5 32
#define OUT_HEIGHT_5 155
#define OUT_WIDTH_5 115
#define N_FILT_5 2
#define IN_HEIGHT_6 155
#define IN_WIDTH_6 115
#define N_CHAN_6 32
#define OUT_HEIGHT_6 155
#define OUT_WIDTH_6 115
#define N_FILT_6 4

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> layer1_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer6_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::conv2d_config {
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const unsigned in_height = IN_HEIGHT_1;
        static const unsigned in_width = IN_WIDTH_1;
        static const unsigned n_chan = N_CHAN_1;
        static const unsigned filt_height = 3;
        static const unsigned filt_width = 3;
        static const unsigned n_filt = N_FILT_1;
        static const unsigned stride_height = 1;
        static const unsigned stride_width = 1;
        static const unsigned out_height = OUT_HEIGHT_1;
        static const unsigned out_width = OUT_WIDTH_1;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct PReLU_config1 : nnet::activ_config {
        static const unsigned n_in = OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config2 : nnet::pooling2d_config {
        static const unsigned in_height = IN_HEIGHT_2;
        static const unsigned in_width = IN_WIDTH_2;
        static const unsigned n_filt = N_FILT_2;
        static const unsigned stride_height = 2;
        static const unsigned stride_width = 2;
        static const unsigned pool_height = 2;
        static const unsigned pool_width = 2;
        static const unsigned out_height = OUT_HEIGHT_2;
        static const unsigned out_width = OUT_WIDTH_2;
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const nnet::Pool_Op pool_op = nnet::Max;
        static const unsigned reuse = 1;
    };

    struct config3 : nnet::conv2d_config {
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const unsigned in_height = IN_HEIGHT_3;
        static const unsigned in_width = IN_WIDTH_3;
        static const unsigned n_chan = N_CHAN_3;
        static const unsigned filt_height = 3;
        static const unsigned filt_width = 3;
        static const unsigned n_filt = N_FILT_3;
        static const unsigned stride_height = 1;
        static const unsigned stride_width = 1;
        static const unsigned out_height = OUT_HEIGHT_3;
        static const unsigned out_width = OUT_WIDTH_3;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct PReLU_config3 : nnet::activ_config {
        static const unsigned n_in = OUT_HEIGHT_3*OUT_WIDTH_3*N_FILT_3;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config4 : nnet::conv2d_config {
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const unsigned in_height = IN_HEIGHT_4;
        static const unsigned in_width = IN_WIDTH_4;
        static const unsigned n_chan = N_CHAN_4;
        static const unsigned filt_height = 3;
        static const unsigned filt_width = 3;
        static const unsigned n_filt = N_FILT_4;
        static const unsigned stride_height = 1;
        static const unsigned stride_width = 1;
        static const unsigned out_height = OUT_HEIGHT_4;
        static const unsigned out_width = OUT_WIDTH_4;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct PReLU_config4 : nnet::activ_config {
        static const unsigned n_in = OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config5 : nnet::conv2d_config {
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const unsigned in_height = IN_HEIGHT_5;
        static const unsigned in_width = IN_WIDTH_5;
        static const unsigned n_chan = N_CHAN_5;
        static const unsigned filt_height = 1;
        static const unsigned filt_width = 1;
        static const unsigned n_filt = N_FILT_5;
        static const unsigned stride_height = 1;
        static const unsigned stride_width = 1;
        static const unsigned out_height = OUT_HEIGHT_5;
        static const unsigned out_width = OUT_WIDTH_5;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config5 : nnet::activ_config {
        static const unsigned n_in = OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };
struct config6 : nnet::conv2d_config {
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const unsigned in_height = IN_HEIGHT_6;
        static const unsigned in_width = IN_WIDTH_6;
        static const unsigned n_chan = N_CHAN_6;
        static const unsigned filt_height = 1;
        static const unsigned filt_width = 1;
        static const unsigned n_filt = N_FILT_6;
        static const unsigned stride_height = 1;
        static const unsigned stride_width = 1;
        static const unsigned out_height = OUT_HEIGHT_6;
        static const unsigned out_width = OUT_WIDTH_6;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct linear_config6 : nnet::activ_config {
        static const unsigned n_in = OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
        };

#endif 
