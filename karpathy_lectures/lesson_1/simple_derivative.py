import math
import numpy as np
import matplotlib.pyplot as plt

h = 0.001
a = 2
b = 3
c = -3

d1 = b*a + c
a += h
d2 = b*a + c

print("d1 = ", d1)
print("d2 = ", d2)
print("slope = ", (d2 - d1)/h)