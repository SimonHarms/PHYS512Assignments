import numpy as np

func1 = np.exp
func2 = lambda x : np.exp(0.1 * x)
epsilon = np.power(2.0,-53)
dx = np.power(35.0/4 * epsilon, 1.0/5)
x = 1
df1 = (16*(func1(x+dx) - func1(x-dx)) - (func1(x+2*dx) - func1(x-2*dx)))/(28*dx)
df2 = (16*(func2(x+dx) - func2(x-dx)) - (func2(x+2*dx) - func2(x-2*dx)))/(28*dx)
print(df1, df1 - func1(x))
print(df2, df2 - 0.1 * func2(x))
