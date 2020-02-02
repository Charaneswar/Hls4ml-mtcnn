import numpy as np
import yaml
import h5py
import json

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
    # print(layer['name'])
    for config, config_value in keras_layer["config"].items():
        if config == "activation":
            layer['activation'] = config_value
            # layer['name'] = keras_layer['name']

            # print(layer['activation'])
    layer['name'] = keras_layer['name']
    print(layer['name'])
    print(keras_layer["config"].items())

# List all groups
# print("Keys: %s" % f.keys())
# a_group_key = list(f.keys())[0]
# print(a_group_key)
# Get the data
# s = list(h5py.AttributeManager.keys(f))
# data = list(f[a_group_key]
print(list(f.attrs.keys()))
print(f.attrs['layer_names'])
ju = 'max_pooling2d_7'
# s = f.get('conv2d_13/conv2d_13')
d = f.get('/{}/{}'.format(ju, ju))
# e = list(d.items())
# r = np.array(d.get('kernel:0'))
print(d)
