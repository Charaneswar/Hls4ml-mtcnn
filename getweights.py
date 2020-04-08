import numpy as np
from mtcnn.network.factory import NetworkFactory
import tensorflow as tf

from numpy import save
from numpy import array, float32
import pandas
#
# p_value = NetworkFactory().build_pnet()
# o_value = NetworkFactory().build_onet()
# r_value = NetworkFactory().build_rnet()

a = np.load('mtcnn/data/mtcnn_weights.npy', allow_pickle=True).tolist()
c = a['pnet'][1]
print(c)

# np.savetxt('p.csv', c, fmt='%s', delimiter=',')