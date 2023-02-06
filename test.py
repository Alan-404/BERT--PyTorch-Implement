#%%
import numpy as np
# %%
a = np.array([1,2,3,4,4,5])
# %%
lst = []

# %%
lst.append(a)
# %%
samples = 5
# %%
inputs = []
labels = []

# %%
import random
for i in range(len(lst)):
    for _ in range(samples):
        temp = lst[i]
        index = random.randint(0, len(temp)-1)
        temp[index] = 100
        inputs.append(temp)
        labels.append(lst[i])
# %%
inputs
# %%

# %%

# %%
inputs
# %%
