import tvm
from hdsl.parser import mybuild
from hdsl.visitor import *

def extern1(a):
  a = a + 2
  return a + 1

def extern2(a):
  _sum = 0
  for x in range(0, 5):
    if _sum%2 == 0:
      _sum = _sum + x
  return _sum

def extern3(a):
  for i in range(0, 5):
    a[i] = a[i] + 1

def test_assign():
  a = 2

def test_tvm_var():
  a = tvm.var("a", dtype="int32")

def test_tvm_placeholder():
  a = tvm.placeholder((2,), name = "A")

def test_tvm_compute():
  A = tvm.placeholder((2,2), name = "A")
  B = tvm.compute(A.shape, lambda x, y: A[x][y] + 1)

def test_assign():
  A = tvm.placeholder((5,), name = "A")
  A[3] = 5

def test_for():
  A = tvm.placeholder((5,), name = "A")
  for i in range(0, 5):
    A[i] = 22

def test_extern1():
  A = tvm.placeholder((2,2), name = "A")
  B = tvm.compute(A.shape, lambda x, y: extern1(A[x][y]))

def test_extern2():
  A = tvm.placeholder((2,2), name = "A")
  B = tvm.compute(A.shape, lambda x, y: extern2(A[x][y]))

def test_extern3():
  A = tvm.placeholder((5,5), name = "A")
  for j in range(0, 5):
    extern3(A[j])

ld = tvm.make.Load("int32", tvm.var("a"), tvm.var("x"), tvm.make.UIntImm("uint1", 1))

v = Visitor().ReplaceVar("x", tvm.var("b"))
stmt = v.mutate(ld)

print stmt

evaluator = mybuild(test_extern3, extern_func = [extern3], args = ["A"])
