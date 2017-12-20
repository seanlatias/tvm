import tvm, inspect, ast, numpy

a = tvm.var("a")
a = a + 1
a = a + 1
print a


A = tvm.placeholder((10,), name = "A", dtype = "int32")
B = tvm.compute(A.shape, lambda x: A[x] + 1, name = "B")
C = tvm.compute(A.shape, lambda x: B[x] + 2, name = "C")
s = tvm.create_schedule(B.op)

l = tvm.lower(s, [A], simple_mode = True)
print l[0].body.condition


Ab = tvm.decl_buffer(A.shape, A.dtype, A.op.name)
Bb = tvm.decl_buffer(A.shape, A.dtype, "B")
ld = tvm.make.Load(A.dtype, Ab.data, 2)
st = tvm.make.Store(Ab.data, ld, 1)
body = B.op.body[0]
axis = B.op.axis[0]
dom = axis.dom
var = axis.var


st2 = tvm.make.Store(Bb.data, tvm.make.Load(A.dtype, Ab.data, var), var)
f = tvm.make.For(var, dom.min, dom.extent, 0, 0, st2)

c = tvm.var("C")
stc = tvm.make.Store(c, 0, 0)
al = tvm.make.Allocate(c, "int32", [10], l[0].body.condition, stc, None)
print al

print f
print tvm.make.Add(ld, 2)
print st
lf = tvm.ir_pass.MakeAPI(al, "test", [Ab], 0, True)
func = tvm.build(lf)
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 1)
arr = numpy.zeros((10), dtype = 'int32')
arr2 = numpy.zeros((10), dtype = 'int32')
arr[1] = 4
arr[2] = 5
arr3 = [0]
inp = tvm.nd.array(arr, tvm.cpu(0))
output = tvm.nd.array(arr2, tvm.cpu(0))
evaluator(inp)
print arr3
