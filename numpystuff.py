import numpy as np

# Create an array of 10 zeros
a = np.zeros(10)
print(a)
# Output: [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

# Create an array of 10 ones
b = np.ones(10)
print(b)
# Output: [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]

# Create an array of 10 random values between 0 and 1
c = np.random.random(10)
print(c)
# Output: [ 0.5488135   0.71518937  0.60276338  0.54488318  0.4236548
#           0.64589411  0.43758721  0.891773    0.96366276  0.38344152]

# Add the arrays a and b
d = a + b
print(d)
# Output: [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]

# Multiply the arrays b and c
e = b * c
print(e)
# Output: [ 0.5488135   0.71518937  0.60276338  0.54488318  0.4236548
#           0.64589411  0.43758721  0.891773    0.96366276  0.38344152]

# Find the mean of the array e
mean = np.mean(e)
print(mean)
# Output: 0.656366830152
