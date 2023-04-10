import numpy as np

A = [
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3]
]

A = np.array(A)
B = [2, 3, 2, 2]
B = np.array(B)

'''
# B = 2**B
A = np.subtract(A, B)

C = [[0.0] * 3] * 4
C = np.array(C)
C[1] = C[1] + np.array([1, 1, 1])
D = np.array([[1] * 3] * 4)
print(C+D)
'''
E = {2:0, 4:0, 3:0}
print(sorted(E.keys()))

A[2] = B
print(A)

