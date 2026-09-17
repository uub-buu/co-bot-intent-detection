import numpy as np

data = np.random.rand(1, 256, 256, 3).astype(np.float32)
data.tofile('dummy_input.raw')