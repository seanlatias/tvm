import tvm, numpy, ast, inspect

"""A Python AST visitor that constructs Halide IR

Member Variables
----------------
io_dict: contains the user-specified IO
         key = name
         value = {'arg': argument to tvm.ir_pass.MakeAPI}
var_dict: contains all tvm.var
          It can be For loop_var, Allocate buffer_var, etc.
          key = name
          value = {'var': tvm.var,
                   'type': {for, tvm, intermediate},
                   'ast': the ast node for an input var,
                   'min': the min_value for a For index var,
                   'extent': the extent for a For index var,
                   'allocated': whether the var is allocated}
buffer_dict: contains all tvm.placeholder
             It can be an input/output or the result from tvm.compute
             key = name
             value = {'tensor': tvm.placeholder,
                      'buffer': the tvm buffer,
                      'shape': the shape of the buffer in tuple,
                      'ast': the ast node for the placeholder,
                      'type': {input, compute},
                      'allocated': whether the buffer is allocated}
externs_dict: the functions that defined outside
              key = name
              value = ast of the extern func
arg_list: a list that contains the input/output names

Member Functions
----------------
Refer to each function for details

enter(): the main entrance for building a Halide IR

visit_body(): visit a list of statements

visit_Name(): visit an AST Name node
"""

class Visitor(ast.NodeVisitor):
  # the general true condition for Allocate node
  true = tvm.make.UIntImm("uint1", 1)

  def enter(self, ast_root, extern_funcs = [], args = [], dfg_root = None):
    """The entry point for the AST Visitor

    Parameters
    ----------
    ast_root: the ast root to be visited, usually an ast.Module node

    extern_funcs: a list of the plain code of extern functions

    args: a list that contains the name of the io variables

    Returns
    -------
    ir: the constructed Halide IR
    """

    # initialize the member variables
    self.io_dict = {}
    self.var_dict = {}
    self.buffer_dict = {}
    self.externs = {}
    for f in extern_funcs:
      self.externs[f] = ast.parse(extern_funcs[f]).body[0]
    self.arg_list = args

    assert(isinstance(ast_root, ast.Module))
    ir = self.visit(ast_root.body[0])
    return ir

  def visit_body(self, nodes):
    """Visit a list of statements in the Python AST"""

    if len(nodes) == 0:
      # create a dummy statement node
      return tvm.make.Evaluate(1)
    else:
      first = nodes[0]
      rest = nodes[1:]
      has_rest = len(rest) > 0
      if (isinstance(node, ast.For) or isinstance(node, ast.If)):
        # imperative code block
        ir_first = self.visit(first)
        if has_rest:
          ir_rest = self.visit_body(rest)
          return tvm.make.Block(ir_first, ir_rest)
        else:
          return ir_first
      elif isinstance(first, ast.Assign):
        targets = first.targets
        value = first.value
        if isinstance(value, ast.Call):
          types = self.check_call_type(value)
          if len(types) == 2 and types[0] == "tvm":
            ir_first = self.visit(first)
            ir = ir_first
            if has_rest:
              ir_rest = self.visit_body(rest)
              ir = tvm.make.Block(ir_first, ir_rest)
            if types[1] == "compute":
              assert len(targets) == 1, "tvm.compute only has one output, instead of " + str(len(targets))
              name = targets[0]
              if name in arg_list:
                return ir
              else:
                assert name in self.buffer_dirct, "undeclared buffer " + name
                buffer = self.buffer_dict[name]
                return tvm.make.Allocate(buffer['buffer'].data, buffer['tensor'].dtype, \
                    buffer['shape'], self.true, ir)
            else: # tvm.var must be in arg_list
              assert len(targets) == 1, "tvm.var only has one output, instead of " + str(len(targets))
              return ir
        else:
          # intermediate variable
          ir_first = self.visit(first)
          assert name in self.var_dict, "undecalred variable " + name
          var = self.var_dict[name]
          ir = ir_first
          if has_rest:
            ir_rest = self.visit_body(rest)
            ir = tvm.make.Block(ir_first, ir_rest)
          if var['allocated']:
            return ir
          else:
            return tvm.make.Allocate(var['var'], var['var'].dtype, [1], self.true, ir)
      else:
        # Not yet supported AST nodes: ClassDef, FunctionDef, Return, Print, While, With, Assert
        return self.visit_body(rest)



    '''
    # the last node
    if len(nodes) == 1:
      if isinstance(nodes[0], ast.For):
        return self.visit(nodes[0])
      elif isinstance(nodes[0], ast.Assign):
        if isinstance(nodes[0].value, ast.Call):
          return self.visit(nodes[0])
        elif isinstance(nodes[0].value, ast.Num):
          name = nodes[0].targets[0].id
          if name in self.var_dict:
            return tvm.make.Store(self.var_dict[name]['var'], nodes[0].value.n, 0)
          else:
            var = tvm.var(name, "int32")
            store = tvm.make.Store(var, nodes[0].value.n, 0)
            return tvm.make.Allocate(var, "int32", [1], true, store)
        elif isinstance(nodes[0].value, ast.IfExp):
          name = nodes[0].targets[0].id
          cond = nodes[0].value.test
          t = nodes[0].value.body
          f = nodes[0].value.orelse
          if isinstance(t, ast.Name):
            t = self.var_dict[nodes[0].value.body.id]['var']
          if isinstance(f, ast.Name):
            f = self.var_dict[nodes[0].value.orelse.id]
            if f['ast'] == None:
              f = tvm.make.Load(f['var'].dtype, f['var'], 0)
          if isinstance(cond, ast.Compare):
            op = cond.ops[0]
            lhs = self.visit(cond.left)
            rhs = cond.comparators[0]
            if isinstance(rhs, ast.Name):
              rhs = self.var_dict[rhs.id]
              if rhs['ast'] == None:
                rhs = tvm.make.Load(rhs['var'].dtype, rhs['var'], 0)
            if isinstance(cond.ops[0], ast.Gt):
              cond = tvm.make.GT(lhs, rhs)
          t = tvm.make.Cast(f.dtype, t)
          sel = tvm.make.Select(cond, t, f)
          var = self.var_dict[name]['var']
          store = tvm.make.Store(var, sel, 0)
          return store
        elif isinstance(nodes[0].value, ast.Subscript):
          rhs = self.visit(nodes[0].value)
          node = nodes[0].targets[0]
          name = node.slice.value.id
          var2 = self.var_dict[name]
          ld = tvm.make.Load(var2['var'].dtype, var2['var'], 0)
          name = node.value.slice.value.id
          var1 = self.var_dict[name]
          name = node.value.value.id
          buff = self.buffer_dict[name]['buffer']
          store = tvm.make.Store(buff.data, rhs, var1['var'] * self.buffer_dict[name]['shape'][1] + ld)
          return store

      elif isinstance(nodes[0], ast.If):
        cond = nodes[0].test
        t = self.visit_body(nodes[0].body)
        f = self.visit_body(nodes[0].orelse)
        if isinstance(cond, ast.Compare):
          op = cond.ops[0]
          lhs = self.visit(cond.left)
          rhs = cond.comparators[0]
          if isinstance(rhs, ast.Name):
            rhs = self.var_dict[rhs.id]
            if rhs['ast'] == None:
              rhs = tvm.make.Load(rhs['var'].dtype, rhs['var'], 0)
          if isinstance(cond.ops[0], ast.Lt):
            cond = tvm.make.LT(lhs, rhs)
        ite = tvm.make.IfThenElse(cond, t, f)
        print ite
        return ite


    elif len(nodes) == 0:
      dum = tvm.var('dum')
      store = tvm.make.Store(dum, 0, 0)
      al = tvm.make.Allocate(dum, "int32", [1], self.true, store)
      return al
    else:
      if isinstance(nodes[0], ast.For):
        first = self.visit(nodes[0])
        rest = self.visit_body(nodes[1:])
        return tvm.make.Block(first, rest)
      elif isinstance(nodes[0], ast.Assign):
        name = nodes[0].targets[0].id
        if isinstance(nodes[0].value, ast.Call):
          first = self.visit(nodes[0])
          rest = self.visit_body(nodes[1:])
          if first == None:
            return rest
          else:
            if name in self.buffer_dict:
              var = self.buffer_dict[name]['buffer'].data
              shape = self.buffer_dict[name]['shape']
              return tvm.make.Allocate(var, "int32", [shape[0], shape[1]], self.true, tvm.make.Block(first, rest))
            else:
              return tvm.make.Block(first, rest)
        elif isinstance(nodes[0].value, ast.Num):
          if name in self.var_dict:
            first = tvm.make.Store(self.var_dict[name]['var'], nodes[0].value.n, 0)
            rest = self.visit_body(nodes[1:])
            return tvm.make.Block(first, rest)
          else:
            var = tvm.var(name, "int32")
            store = tvm.make.Store(var, nodes[0].value.n, 0)
            self.var_dict[name] = {'var': var, 'ast': None}
            rest = self.visit_body(nodes[1:])
            return tvm.make.Allocate(var, "int32", [1], self.true, tvm.make.Block(store, rest))
    '''

  """ Main Visitors """

  def visit_FunctionDef(self, node):
    ir = self.visit_body(node.body)
    print ir
    buff = []
    for n in self.io_dict:
      buff.append(self.io_dict[n]['arg'])
    api = tvm.ir_pass.MakeAPI(ir, 'IR', buff, 0, True)
    func = tvm.build(api)
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 1)
    from digitrec_data import read_digitrec_data
    train_data, train_label, test_data, test_label = read_digitrec_data()
    arr = numpy.zeros((10, 100), dtype = 'int32')
    arr2 = numpy.zeros((10, 3), dtype = 'int32')
    for i in range(0, 10):
      for j in range(0, 100):
        arr[i][j] = int(''.join(str(int(e)) for e in train_data[i][j]), 2)
    input_image = int(''.join(str(int(e)) for e in test_data[0]), 2)
    label_val = tvm.nd.array(arr, tvm.cpu(0))
    knn_mat = tvm.nd.array(arr2, tvm.cpu(0))
    evaluator(input_image, knn_mat, label_val)
    print knn_mat

  def visit_Subscript(self, node):
    """Visit A[x] or A[x][y]

    Returns
    -------
    Expr: x or x * extent + y
    """
    if isinstance(node.value, ast.Subscript):
      # a 2D array
      var2 = node.slice.value
      if isinstance(var2, ast.Name):
        var2 = self.var_dict[var2.id]['var']
      else
        var2 = var2.n
      var1 = node.value.slice.value
      if isinstance(var1, ast.Name):
        var1 = self.var_dict[var1.id]['var']
      else
        var1 = var1.n
      buffer_name = node.value.value.id
      buffer_shape = self.buffer_dict[buffer_name]['shape']
      return var1 * buffer_shape[1] + var2
    else:
      var = node.slice.value
      if isinstance(var, ast.Name):
        var = self.var_dict[var.id]['var']
      else:
        var = var.n
      return var

  def visit_For(self, node):
    """Visit for i in range(a, b)

    Returns
    -------
    Stmt: For node
    """
    index = node.target.id
    min_val = node.iter.args[0].n
    max_val = node.iter.args[1].n
    extent = max_val - min_val
    var = tvm.var(index)
    self.var_dict[index] = {'var': var, 'type': 'for', 'min': min_val, 'extent': extent, 'allocated': True}
    stmt = self.visit_body(node.body)
    return tvm.make.For(var, min_val, extent, 0, 0, stmt)

  def visit_Assign(self, node):
    """Visit targets = value

    Returns
    -------
    Stmt: Store node or tvm.compute IR
    """
    # Currently, we only allow one output target
    target = node.targets[0]
    index = 0
    content = None
    is_tvm = False
    dtype = "float32"


    # Analyze right hand side first
    if isinstance(node.value, ast.Call):
      call = node.value
      call_type = check_call_type(call)
      if len(call_type) == 1:
        # External function call. We do not support it right now
        content = self.visit(call)
      else:
        args = call.args
        keywords = call.keywords
        # Currently we only support tvm calls
        if call_type[0] == "tvm":
          is_tvm = True
          if call_type[1] == "var": # tvm.var
            assert(isinstance(target, ast.Name)), "target of tvm.var must be a name"
            for keyword in keywords: # check every keyword in tvm.var
              if keyword[0] == "dtype":
                dtype = keyword[1].id
              elif keyword[0] == "name":
                pass
              else:
                raise ValueError("Unknown/Unsupported keyowrds to tvm.var: " + str(keyword[0]))
            name = target.id
            var = tvm.var(name, dtype = dtype)
            self.var_dict[name] = {'var': var, 'tpye': 'tvm', 'allocated': True}
            if name in self.arg_list: # check whether this var belongs to io
              self.io_dict[name] = {'arg': var}
          elif call_type[1] == "placeholder": # tvm.placeholder
            assert(isinstance(target, ast.Name)), "target of tvm.placeholder must be a name"
            for keyword in keywords: # check every keyword in tvm.var
              if keyword[0] == "dtype":
                dtype = keyword[1].id
              elif keyword[0] == "name":
                pass
              else:
                raise ValueError("Unknown/Unsupported keyowrds to tvm.var: " + str(keyword[0]))

        else:
          raise ValueError("Currently we only support tvm functions")

      func = node.value.func
      if (func.value.id == "tvm"):
        if (func.attr == "var"): # a tvm variable
          name = targets[0].id
          var = tvm.var(name, "int32")
          self.var_dict[name] = {'var': var, 'ast': node, 'input': True}
          if name in self.arg_list:
            self.io_dict[name]= {'arg': var}
        elif (func.attr == "placeholder"):
          name = targets[0].id
          shape = node.value.args[0].elts
          length = len(shape)
          if length == 1:
            pass
          elif length == 2:
            placeholder = tvm.placeholder((shape[0].n, shape[1].n), name = name, dtype = "int32")
            buff = tvm.decl_buffer(placeholder.shape, placeholder.dtype, placeholder.name)
            self.buffer_dict[name] = {'tensor': placeholder, 'buffer': buff, 'ast': node, 'shape': (shape[0].n, shape[1].n)}
          if name in self.arg_list:
            self.io_dict[name] = {'arg': buff}
        elif (func.attr == "compute"):
          name = targets[0].id
          shape_ast = None
          shape = None
          if isinstance(node.value.args[0], ast.Attribute):
            assert(node.value.args[0].attr == "shape")
            n = node.value.args[0].value.id
            shape = self.buffer_dict[n]['tensor'].shape
            tup = ast.Tuple([ast.Num(shape[0]), ast.Num(shape[1])], ast.Load())
            shape_ast = tup
          else:
            shape = node.value.args[0].elts
            shape = (shape[0].n, shape[1].n)
            shape_ast = node.value.args[0]
          lamb = node.value.args[1]
          n = ast.Name("_internal", ast.Store())
          c = ast.Call(func, [shape_ast, lamb], [], None, None)
          ass = ast.Assign([n], c)
          stmt = []
          for v in self.var_dict:
            if self.var_dict[v]['ast'] != None:
              stmt.append(self.var_dict[v]['ast'])
          for b in self.buffer_dict:
            stmt.append(self.buffer_dict[b]['ast'])
          for f in self.externs:
            stmt.append(self.externs[f])
          stmt.append(ass)
          stmt = ast.Module(stmt)
          ast.fix_missing_locations(stmt)

          exec(compile(stmt, '<ast>', 'exec'), globals(), globals())

          placeholder = tvm.placeholder(shape, name = name, dtype = "int32")
          buff = tvm.decl_buffer(placeholder.shape, placeholder.dtype, placeholder.name)
          f = func
          f.attr = "placeholder"
          c = ast.Call(f, [shape_ast], node.value.keywords, None, None)
          ass = ast.Assign(targets, c)
          self.buffer_dict[name] = {'tensor': placeholder, 'buffer': buff, 'ast': ass, 'shape': shape}

          self.axis = _internal.op.axis
          expr = self.replace_call_with_load( _internal.op.body[0])
          ir = None
          if len(self.axis) == 1:
            index = self.axis[0].var
            store = tvm.make.Store(buff.data, expr, index)
            ir = tvm.make.For(self.axis[0].var, self.axis[0].dom.min, self.axis[0].dom.extent, 0, 0, store)
          else:
            index = self.axis[0].var * self.axis[1].dom.extent + self.axis[1].var
            store = tvm.make.Store(buff.data, expr, index)
            ir = tvm.make.For(self.axis[1].var, self.axis[1].dom.min, self.axis[1].dom.extent, 0, 0, store)
            ir = tvm.make.For(self.axis[0].var, self.axis[0].dom.min, self.axis[0].dom.extent, 0, 0, ir)
          print ir
          return ir

    else:
      ir = self.visit(node.value)

      pass
    # left hand side
    if isinstance(target, ast.Name):
      target = target.id
      if target in self.var_dict:
        target = self.var_dict[target]['var']
      else:
        var = tvm.var(target)
        self.var_dict[target] = {'var': var, 'type': 'intermediate', 'allocated': False}
        target = var
    else:
      assert isinstance(target, ast.Subscript)
      index = self.visit(target)
      if isinstance(target.value, ast.Subscript): # 2D
        target = target.value.value.id
      else                                        # 1D
        target = target.value.id
      assert target in self.buffer_dict, "undeclared buffer " + target
      target = self.buffer_dict[target]['buffer'].data


  """Helper Functions"""

  def check_call_type(self, node):
    """Check the type of ast.Call

    It could be tvm.xxx or just a simple function

    Returns
    -------
    [module_name, function_name] or [function_name]
    """
    assert isinstance(node, ast.Call), "the input should be ast.Call"
    if isinstance(node.func, ast.Attribute):
      return [node.func.value.id, node.func.attr]
    else:
      return [node.func.id]


  def replace_call_with_load(self, node):
    if isinstance(node, tvm.expr.Call):
      if node.call_type == 5:
        exprs = []
        for arg in node.args:
          exprs.append(self.replace_call_with_load(arg))
        call = tvm.make.Call("int32", node.name, exprs, node.call_type, node.func, node.value_index)
        return call
      elif node.call_type == 3:
        buff = self.buffer_dict[node.name]['buffer']
        axis = self.axis
        if len(axis) == 1:
          load = tvm.make.Load(node.dtype, buff.data, axis[0].var)
        else:
          load = tvm.make.Load("int32", buff.data, axis[0].var * axis[1].dom.extent + axis[1].var)
        return load
    elif isinstance(node, tvm.expr.Var):
      var = self.var_dict[node.name]['var']
      return var
    elif isinstance(node, tvm.expr.Sub):
      a = self.replace_call_with_load(node.a)
      b = self.replace_call_with_load(node.b)
      return tvm.make.Sub(a, b)
    elif isinstance(node, tvm.expr.Mul):
      a = self.replace_call_with_load(node.a)
      b = self.replace_call_with_load(node.b)
      return tvm.make.Mul(a, tvm.make.Cast("int32", b))
    elif isinstance(node, tvm.expr.Add):
      a = self.replace_call_with_load(node.a)
      b = self.replace_call_with_load(node.b)
      return tvm.make.Add(a, b)
    else:
      return node

