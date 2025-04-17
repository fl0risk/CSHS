import numpy as np

A = np.ones(2*3*1*6).reshape((2,3,1,6))

A[:,:,0,:] = 3* np.ones(2*3*6).reshape((2,3,6))
A[0,0,0,0] = 5
A[0,1,0,0] = 7
A[0,2,0,4] = 6
A[1,1,0,0] = 192
A[1,2,0,4] = 100
print('Matrix', A)
print(A[0,:,0,:])
print(A[1,:,0,:])
print(np.max(A, axis = (0,2)))