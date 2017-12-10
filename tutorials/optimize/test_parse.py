"""
How to optimize GEMM on CPU
===========================
**Author**: `Jian Weng <https://github.com/were>`_

(TL;DR) TVM provides abstract interfaces which allows users to depict an algorithm and the
algorithm's implementing organization (the so-called schedule) separately. Typically, writing
algorithm in high-performance schedule breaks the algorithm's readability and modularity. Also,
trying various seemingly promising schedules is time-consuming. With the help of TVM, we can
try these schedules efficiently to enhance the performance.

In this tutorial, we will demonstrate how square matrix multiplication is optimized step by step by
writing TVM.

There are two important optmizations on intense computation applications executed on CPU:
    1. Increase the cache hit rate of memory access. Both complex numerical computation and hot-spot
       memory access can be acclerated from high cache hit rate. This requires us to transform the
       origin memory access pattern to the pattern fits the cache policy.
    2. SIMD (Single instruction multi-data), or we call it vector processing unit. Every time, a
       small batch of data, rather than a single grid, will be processed. This requires us to
       transform the data access pattern in the loop body in uniform pattern so that the LLVM
       backend can lower it to SIMD.

Actually, all the methodologies used in this tutorial is a subset of tricks mentioned in this
`repo <https://github.com/flame/how-to-optimize-gemm>`_. Some of them have been applied by TVM
abstraction automatically, but some of them cannot be simply applied due to TVM constraints.

All the experiment results mentioned below, are executed on 2013's 15' MacBook equiped with
Intel i7-2760QM CPU. The cache line size should be 64 bytes for all the x86 CPU.
"""

###############################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial we assume all the matrix tensors are square and fix-bounded.
# We use 1024x1024 float32 matrix in demonstration. Before actually demonstrating,
# we first define these variables. Then we write a baseline implementation,
# the simplest way to write a matrix mulplication in TVM.
#

import tvm
import tvm.schedule as schedule
import tvm.ir_pass as ir_pass
import numpy
import time
import inspect as ip
import ast, re

def myfun():
# The size of the square matrix
  N = 1024
  N = N + 1
# The default tensor type in tvm
  dtype = "float32"
# Random generated tensor for testing
  a = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
  b = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
# The expected answer
  answer = numpy.dot(a.asnumpy(), b.asnumpy())

# Algorithm
  k = tvm.reduce_axis((0, N), 'k')
  A = tvm.placeholder((N, N), name = 'A')
  B = tvm.placeholder((N, N), name = 'B')
  V = tvm.var('V')
  cond = True
  if cond:
    C = tvm.compute(
               A.shape,
               lambda x, y: tvm.sum(A[1, k] * B[k, y] * V, axis = k),
               name = 'C')
  else:
    C = tvm.compute(
               A.shape,
               lambda x, y: tvm.sum(A[x, k] + B[k, y], axis = k),
               name = 'C')
  D = tvm.compute(C.shape,
                  lambda x, y: C[x, y] + A[x, y],
                  name = 'D')


# Default schedule
  s = tvm.create_schedule(C.op)
  s2 = tvm.create_schedule(C.op)
  #func = tvm.build(s, [A, B, C], name = 'mmult')
  IR, _, _  = tvm.lower(s2, [A,B,C], name = 'mmult', simple_mode=True)

  print IR
  print IR.body.body.body.rest.body.value.a.index

  #test lowering
  '''
  binds, arg_list = tvm.get_binds([A, B, D], None)
  bounds = schedule.InferBound(s2)
  stmt = schedule.ScheduleOps(s2, bounds)
  print stmt

  stmt = tvm.lower(s2, [A, B], simple_mode=True)
  print dir(stmt.body.body.body.body.first)
  print stmt.body.body.body.body.first.body

  assert func
  evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 1)
  c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
  print('Baseline: %f' % evaluator(a, b, c).mean)
  '''

myfun()

src = ip.getsource(myfun)
src = re.sub(r'#.*\n', "\n",  src)
src = re.sub(r'\'\'\'.*\'\'\'', "\n", src, flags=re.S)

#print src

root = ast.parse(src)

#print root.body[0].body
#print dir(myfun.func_code)

class myVisitor(ast.NodeVisitor):

  def __init__(self):
    self.axes = []
    self.placeholders = []
    self.computes = []
    self.variables = {}

  def enter(self, node):
    self.scope = {}
    self.visit(node)
    #print self.variables
    return self.placeholders, self.computes, self.axes

  def visit_Assign(self, node):
    targets = node.targets
    if (isinstance(node.value, ast.Call)):
      call = node.value
      func = call.func
      if (isinstance(func.value, ast.Name)):
        module = str(func.value.id)
        name = str(func.attr)
        if module == "tvm":
          if name == "placeholder":
            self.placeholders.append(node)
          elif name == "compute":
            self.computes.append(node)
          elif name == "reduce_axis":
            self.axes.append(node)
    else:
      assert (len(targets) == 1)
      assert (isinstance(targets[0], ast.Name))
      name = str(targets[0].id)
      if name in self.variables:
        self.variables[name].append(node)
      else:
        self.variables[name] = [node]


stmts = ast.parse("N=10").body
stmts2 = ast.parse("N=10").body

#print ip.getsourcelines(myfun)
pl, com, ax =  myVisitor().enter(root)

for p in pl:
  stmts.append(p)
  stmts2.append(p)
for a in ax:
  stmts.append(a)
  stmts2.append(a)
stmts.append(com[0])
stmts2.append(com[1])

s = ast.parse('s = tvm.create_schedule(C.op)')
lower = ast.parse('expr, binds, arg_list= tvm.lower(s, [A, B, C], simple_mode=True)')
lower2 = ast.parse('expr2, binds2, arg_list2 = tvm.lower(s, [A, B, C], simple_mode=True)')

stmts = stmts + s.body + lower.body
stmts2 = stmts2 + s.body + lower2.body

node = ast.Module(stmts)
node2 = ast.Module(stmts2)
ast.fix_missing_locations(node)
ast.fix_missing_locations(node2)

exec(compile(node, '<ast>', 'exec'))
args = [A, B, C]
exec(compile(node2, '<ast>', 'exec'))
args2 = [A, B, C]

'''
# C
print dir(expr.body.body.body.first.buffer_var)
print expr.body.body.body.rest.body.buffer_var
print expr.body.body.body.rest.body.value.a.buffer_var
# A
print expr.body.body.body.rest.body.value.b.a.buffer_var
# B
print expr2.body.body.body.rest.body.value.b.b.buffer_var
'''

expr2.body.body.body.first.buffer_var = expr.body.body.body.first.buffer_var
expr2.body.body.body.rest.body.buffer_var = expr.body.body.body.rest.body.buffer_var
expr2.body.body.body.rest.body.value.a.buffer_var = expr.body.body.body.rest.body.value.a.buffer_var
expr2.body.body.body.rest.body.value.b.a.buffer_var = expr.body.body.body.rest.body.value.b.a.buffer_var
expr2.body.body.body.rest.body.value.b.b.buffer_var = expr.body.body.body.rest.body.value.b.b.buffer_var


cond = tvm.var('cond')
stmt = tvm.make.IfThenElse(tvm.make.EQ(cond, 1), expr, expr2)
stmt = tvm.make.LetStmt(expr2.body.body.body.rest.body.value.b.b.buffer_var, expr.body.body.body.rest.body.value.b.b.buffer_var, stmt)
stmt = tvm.make.LetStmt(expr2.body.body.body.rest.body.value.b.a.buffer_var, expr.body.body.body.rest.body.value.b.a.buffer_var, stmt)
stmt = tvm.make.LetStmt(expr2.body.body.body.rest.body.value.a.buffer_var, expr.body.body.body.rest.body.value.a.buffer_var, stmt)

print expr2.body.body.body.rest.body.value.b.b.index

import HDSL.tensor as t

X = t.Tensor(A)
X[2] = 5
print X.ndim

binds3, arg_list3 = tvm.get_binds([cond], binds2)
func = ir_pass.MakeAPI(stmt, "mystmt", arg_list3 + arg_list, 0, True)
func = tvm.build(func)
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 1)
dtype = "float32"
a = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
b = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
c1 = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
c2 = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
c3 = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
evaluator(1, a, b, c1)
evaluator(0, a, b, c2)
evaluator(-1, a, b, c3)
#print c1
#print c2
#print c3
assert func

#print mystmt

'''
################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunck will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# meomry locality. In this tutorial, I pick up 8, a relatively small value (8 ints < 64 bytes),
# as the blocking size.
#

bn = 8
# Blocking by loop tiling
yo, xo, yi, xi = s[C].tile(C.op.axis[1], C.op.axis[0], bn, bn)
# Hoist reduction domain outside the blocking loop
s[C].reorder(yo, xo, k, yi, xi)
func = tvm.build(s, [A, B, C], name = 'mmult')
assert func
# By simply tiling the loop 8x8, and hoisting k outside the blocking loops, we can get nearly 4x
# speedup compared with the baseline.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt1: %f' % evaluator(a, b, c).mean)

###################################################################################################
# Vectorization
# -------------
# Another important trick is vectorization. When the memory access pattern is uniform, the compiler
# can detect this pattern and pass the continuous memory to vector processor. In TVM, we can use
# `vectorize` interface to hint the compiler this pattern, so that we can accelerate it vastly.
#

# After trying different schedule, we finally found that we can benefit from vectorizing
# the row loop most, i.e. yi.
s[C].vectorize(yi)
func = tvm.build(s, [A, B, C], name = 'mmult')
assert func
# We can get almost another 4x speedup compared with the previous schedule.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt2: %f' % evaluator(a, b, c).mean)

###################################################################################################
# Array Packing
# -------------
# Another important trick is array packing. This trick is to reorder the storage dimension of the
# array to convert the continuous access pattern on certain dimension to a sequential pattern after
# flattening.
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/array-packing.png
#      :align: center
#      :scale: 100%
#


###################################################################################################
# Just as it is shown in the figure above, after blocking the computations, we can observe the array
# access pattern of B (after flattening), which is regular but discontinuous. We expect that after
# some transformation we can get continuous access pattern. We can reorder a [16][16] array to
# a [16/4][16][4] array, so that the access pattern of B will be sequential when grabing
# the corresponding value from the packed array.
#

# We have to re-write the algorithm slightly.
packedB = tvm.compute((N / bn, N, bn), lambda x, y, z: B[y, x * bn + z], name = 'packedB')
C = tvm.compute(A.shape,
                lambda x, y: tvm.sum(A[x, k] * packedB[y / bn, k, y % bn], axis = k),
                name = 'C')

# Same schedule
s = tvm.create_schedule(C.op)
yo, xo, yi, xi = s[C].tile(C.op.axis[1], C.op.axis[0], bn, bn)
s[C].reorder(yo, xo, k, yi, xi)
s[C].vectorize(yi)

func = tvm.build(s, [A, B, C], name = 'mmult')
assert func
# We can accelerate it almost 3x compared with the previous schedule.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt3: %f' % evaluator(a, b, c).mean)

##################################################################################################
# Summary
# -------
# After applying three main tricks, we can achieve almost 90% performance of numpy.
# Further observation is required to catch up with the performance of numpy.
#

# TODO(Jian Weng): Catch up with the performance of numpy.
_a = a.asnumpy()
_b = b.asnumpy()
now = time.clock()
answer = numpy.dot(_a, _b)
print("Numpy: %f" % (time.clock() - now))
'''
