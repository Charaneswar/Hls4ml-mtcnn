import numpy as np
import yaml
import h5py
import json


# //////////

def print_array_to_cpp(name, a, odir):
    # count zeros
    zero_ctr = 0
    for x in np.nditer(a, order='C'):
        if x == 0:
            zero_ctr += 1

    # put output in subdir for tarballing later
    f = open("{}/firmware/weights/{}.h".format(odir, name), "w")

    # meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {}\n".format(np.min(a)))
    f.write("//Max {}\n".format(np.max(a)))
    f.write("//Number of zeros {}\n".format(zero_ctr))
    f.write("\n")

    # c++ variable
    if "w" in name:
        f.write("weight_default_t {}".format(name))
    elif "b" in name:
        f.write("bias_default_t {}".format(name))
    else:
        raise Exception('ERROR: Unkown weights type')

    for x in a.shape:
        f.write("[{}]".format(x))
    f.write(" = {")

    # fill c++ array.
    # not including internal brackets for multidimensional case
    i = 0
    for x in np.nditer(a, order='C'):
        if i == 0:
            f.write("{}".format(x))
        else:
            f.write(", {}".format(x))
        i = i + 1
    f.write("};\n")
    f.close()

    return zero_ctr

# //////////////

# /Users/charan/Desktop/elec.I/Project/Hls4ml-mtcnn/hls4ml2/keras_config.yml
with open('/Users/charan/Desktop/elec.I/Project/Hls4ml-mtcnn/hls4ml2/keras_config.yml') as file:
    yamlConfig = yaml.full_load(file)
with open(yamlConfig['KerasJson']) as json_file:
    model_arch = json.load(json_file)
# skip_layers = ['InputLayer', 'Dropout', 'Flatten']
skip_layers = []
layer_counter = 0
f = h5py.File(yamlConfig['KerasH5'], 'r')
for keras_layer in model_arch["config"]["layers"]:
    if keras_layer["class_name"] in skip_layers:
        continue

    layer_counter = layer_counter + 1

    # Dictionary to fill in and append to layer_list
    layer = {}

    # Extract name for finding weights and biases
    # weights = f['/{}/{}/kernel:0']
    # .format(layer['name'], layer['name'])][()]
    # print(layer['name']]
    layer['name'] = keras_layer['name']
    for config, config_value in keras_layer["config"].items():
        if config == "activation":
            layer['activation'] = config_value
            # ju = 'max_pooling2d_7'
            # s = f.get('conv2d_13/conv2d_13')
            d = f.get('/{}/{}'.format(layer['name'], layer['name']))
            e = list(d.items())
            r = np.array(d.get('kernel:0'))
            l = np.array(d.get('bias:0'))
            print(r.shape)
            print(l.shape)

    # print(layer['name'])
    # print(keras_layer["config"].items())

# List all groups
# print("Keys: %s" % f.keys())
# a_group_key = list(f.keys())[0]
# print(a_group_key)
# Get the data
# s = list(h5py.AttributeManager.keys(f))
# data = list(f[a_group_key]
# print(list(f.attrs.keys()))
# print(f.attrs['layer_names'])
# print(d)