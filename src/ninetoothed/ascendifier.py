import ast


class Ascendifier(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.max_axes = None
        try:
            from triton.backends.ascend.runtime.utils import valid_axis_names
            self.max_axes = len(valid_axis_names)
        except ImportError:
            pass
    
    def visit_Attribute(self, node):
        self.generic_visit(node)

        if isinstance(node.value, ast.Name) and node.value.id == "tl":
            if node.attr == "float64":
                node.attr = "float32"
        
        return node
      
    def visit_ImportFrom(self, node):
        self.generic_visit(node)
        
        if node.module == "triton.language.extra":
            for alias in node.names:
                if alias.name == "libdevice":
                    node.module = "triton.language.extra.cann"
                    
        return node
      
    def visit_Call(self, node):
      self.generic_visit(node)
      
      is_autotune = (
          isinstance(node.func, ast.Attribute) and 
          isinstance(node.func.value, ast.Name) and 
          node.func.value.id == "triton" and 
          node.func.attr == "autotune"
      )
      
      if is_autotune:
          for kw in node.keywords:
              if kw.arg == "key":
                  filtered_keys = [
                      elt for elt in kw.value.elts
                      if isinstance(elt, ast.Constant) and "size" in str(elt.value)
                  ][:self.max_axes]
                  kw.value.elts = filtered_keys
                
      is_load = (
          isinstance(node.func, ast.Attribute) and 
          isinstance(node.func.value, ast.Name) and 
          node.func.value.id == "triton" and 
          node.func.value.id == "language" and
          node.func.attr == "load"
      ) or (
          isinstance(node.func, ast.Attribute) and 
          node.func.attr == "load"
      )

      if is_load:
          for kw in node.keywords:
              if kw.arg == "other" and isinstance(kw.value, ast.Constant) and kw.value.value is None:
                  kw.value.value = 0.0
                  
      return node