import tvm, numpy
from hdsl.parser import mybuild

SIZE = 100
K = 3

dtype = "float32"

def popcount(i):
  '''
  m1  = 0x5555555555555555
  m2  = 0x3333333333333333
  m4  = 0x0f0f0f0f0f0f0f0f
  h01 = 0x0101010101010101
  x -= (x >> 1) & m1
  x = (x & m2) + ((x >> 2) & m2)
  x = (x + (x >> 4)) & m4
  return (x * h01) >> 56
  '''
  '''
  i = i - ((i >> 1) & 0x55555555)
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 16843009) >> 24
  '''
  return i


def popcount2(x, length):
  out = 0
  for i in range(0, length):
    if x % 2 == 1:
      out += 1
    x = x >> 1
  return out

def update_knn(dist, knn_row):
  max_id = 0
  max_dist = 0
  for k in range(0, K):
    max_id = k if knn_row[k] > max_dist else max_id
  if dist < max_dist:
    knn_row[max_id] = dist

def digitrec():

  input_image = tvm.var("input_image")
  labelval = tvm.placeholder((10, 100), name = 'labelval')
  diff = tvm.compute(labelval.shape, lambda x, y: input_image ^ labelval[x, y], name = 'diff')
  dist = tvm.compute((10, 100), lambda x, y: popcount(diff[x, y]), name = 'dist')

  knn_mat = tvm.placeholder((10, 3), name = 'knn_mat')
  for i in range(0, 10):
    for j in range(0, 100):
      max_id = 0
      max_dist = 0
      for k in range(0, 3):
        max_id = k if knn_mat[i][k] > max_dist else max_id
      if dist[i][j] < max_dist:
        knn_mat[i][max_id] = dist[i][j]

  #s = tvm.create_schedule(knn_mat.op)

evaluator = mybuild(digitrec, extern_func = [popcount], args = ["input_image", "labelval", "knn_mat"])
'''
data = tvm.nd.array(numpy.random.rand(10, SIZE).astype(dtype), tvm.cpu(0))
output = tvm.nd.array(numpy.zeros((10, K), dtype = dtype), tvm.cpu(0))

evaluator(input_image, data, output)
'''
