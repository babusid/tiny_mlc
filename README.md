# A tiny machine learning compiler

Package layout now follows a standard `src` structure:

```python
from tmlc import Tensor
from tmlc.tensor import add
```

Future subpackages should live under `src/tmlc`, for example `src/tmlc/ops`.
