import tvm, numpy
import hdsl

def extern(x, y):
  x = y * 2
  return x - y

def myfunc():

  A = tvm.placeholder((10, 10), name = "A", dtype = "int32")
  B = tvm.placeholder((10, 10), name = "B", dtype = "int32")

  for i in range(0, 10):
    for j in range(0, 10):
      if A[i][j] != B[i][j]:
        B[i][j] = A[i][j]

  C = tvm.compute((10, 10), lambda x, y: extern(A[x, y], B[x, y]),
      name = "C", dtype = "int32")

  D = tvm.placeholder((10, 10), name = "D", dtype = "int32")
  for i in range(0, 10):
    for j in range(0, 10):
      D[i][j] = 0 if C[i][j] < 0 else C[i][j]

if __name__ == "__main__":
  in_arr1 = np.zeros((10, 10), dtype = "int32")
  in_arr2 = np.array(...)
  out_arr = np.zeros((10, 10), dtype = "int32")

  evaluator = hdsl.build(myfunc, tvm.cpu(0), extern_funcs = [extern],
      inputs = ["A", "B"], outputs = ["C"])

  in_arr1 = in_arr2 * 2

  tvm_in_arr1 = tvm.nd.array(in_arr1, tvm.cpu(0))
  tvm_in_arr2 = tvm.nd.array(in_arr2, tvm.cpu(0))
  tvm_out_arr = tvm.nd.array(out_arr, tvm.cpu(0))
  evaluator(tvm_in_arr1, tvm_in_arr2, tvm_out_arr)

  out_arr = tvm_out_arr.asnumpy()

  out_arr = out_arr + 2



