import numpy as np

data = np.transpose(np.loadtxt("dish_zenith.txt"))
x = data[0]
y = data[1]
z = data[2]

ndata = x.size
A = np.zeros([ndata,4])
A[:,0] = x**2 + y**2
A[:,1] = x
A[:,2] = y
A[:,3] = np.ones([1,ndata])
A = np.matrix(A)
d = np.matrix(z).transpose()
lhs = A.transpose()*A
rhs = A.transpose()*d
param = np.linalg.inv(lhs)*rhs
a = param[0,0]
x0 = -param[1,0]/(2*a)
y0 = -param[2,0]/(2*a)
z0 = param[3,0] - a * (x0**2 + y0**2)
print("a: ", a, ", x0: ", x0, ", y0: ", y0, ", z0: ", z0)
err_mat = d - A*param
var_estimate = ((err_mat.transpose()*err_mat)/(ndata - 4))[0,0]
coeff_variance = var_estimate * np.linalg.pinv(lhs)
a_err = coeff_variance[0,0]
print("error in a: ", a_err)
print("Focal length is ", 1/(4*a), ", with error: ", (1/(2*a**2))**2 * a_err)
