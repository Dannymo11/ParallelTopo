"""CPU baseline simulator (M1).

NumPy/SciPy implementation of the slim transit simulator. Exposes the canonical
``reset(seed) -> state``, ``step(state, action) -> (state, reward, done, info)``
API that the GPU port (``sim_gpu``) mirrors at batch dimension 1.
"""
