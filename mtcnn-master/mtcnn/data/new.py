import numpy as np
# import weigh
from numpy import save
from numpy import array, float32
import pandas

#
# data = np.array([[0.02040549, -0.0032297, 0.01252353, ..., -0.02456038, -0.07111277, -0.04677726],
#               [-0.06263377, 0.06706491, 0.00028469, ..., 0.00287913, 0.00133757, 0.00738636],
#               [0.03862411, -0.04655068, -0.02518571, ..., -0.06601012, -0.08898577, -0.09306277], ...,
#               [-0.00080055, -0.01099595, -0.02862552, ..., 0.05072279, 0.02052492, 0.03238122],
#               [0.06329422, 0.03945398, -0.00575098, ..., -0.00520972, -0.08758462, 0.00224501],
#               [-0.0223851, 0.00024065, 0.00757738, ..., 0.00446005, 0.00156003, -0.02281255]])
# f = open("weights2.txt", "r")
# a =np.load('mtcnn_weights.npy')
a = np.load('mtcnn_weights.npy', allow_pickle=True).tolist()
# b = [1]
# c = np.hstack((a, b))
# save("foo.csv", a,)
# a=f.read()
# np.savetxt('array.txt', c, delimiter=',')
# save("new",content)
c = a['pnet']
print(c)
np.savetxt('p.csv', c, fmt='%s', delimiter=',')