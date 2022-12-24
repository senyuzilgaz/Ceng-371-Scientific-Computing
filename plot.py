import numpy as np
import matplotlib.pyplot as plt
import crout
import sherman
import time

def hilbert(m):
  a = np.zeros ( ( m, m ) )
  for i in range ( 0, m ):
    for j in range ( 0, m ):
      a[i,j] = 1.0 / float ( i + j + 1 )
  return a

times_sherman =[]
times_crout =[]
errors_sherman = []
errors_crout = []
sizes = range(1, 301)

for size in sizes:

  A = hilbert(size)

 
  start_time_crout = time.time()
  L, U = crout.recursive_crout(A)
  end_time_crout = time.time() - start_time_crout

  start_time_sherman = time.time()
  L1, U1 = sherman.sherman_factorization_recursive(A)
  end_time_sherman = time.time() - start_time_sherman

  
  error_crout = np.linalg.norm((A - L @ U) / A)
  error_sherman = np.linalg.norm((A - L1 @ U1) / A)

  times_crout.append(end_time_crout)
  times_sherman.append(end_time_sherman)
  errors_crout.append(error_crout)
  errors_sherman.append(error_sherman)

plt.plot(sizes, errors_sherman, label='Shermans Error')
plt.xlabel('matrix size')
plt.ylabel('time')
plt.legend()
plt.show()





