import matplotlib.pyplot as plt
import tensorflow as tf
from src.data.getter.getter import DataGetter
import numpy as np
import random


getter = DataGetter()
records = getter.data
labels = [np.argmax(i.label) for i in records]


# get random 1000 samples
index = random.randint(0, len(labels) - 1000)
labels = labels[index: index + 1000]


data = list(zip(range(len(labels)), labels))

data = tf.unstack(data, axis=1)
data = tf.Session().run(data)

plt.plot(data[0], data[1])
plt.show()
