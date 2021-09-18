import numpy as np

func1 = np.exp
func2 = lambda x : np.exp(0.1 * x)
epsilon = np.power(2.0,-53)
dx1 = np.power(35.0/4 * epsilon, 1.0/5)
dx2 = np.power(35.0/4 * epsilon, 1.0/4) #slightly different value of dx to show that the first one is correct
x = 1
df11 = (16*(func1(x+dx1) - func1(x-dx1)) - (func1(x+2*dx1) - func1(x-2*dx1)))/(28*dx1)
df21 = (16*(func2(x+dx1) - func2(x-dx1)) - (func2(x+2*dx1) - func2(x-2*dx1)))/(28*dx1)
df12 = (16*(func1(x+dx2) - func1(x-dx2)) - (func1(x+2*dx2) - func1(x-2*dx2)))/(28*dx2)
df22 = (16*(func2(x+dx2) - func2(x-dx2)) - (func2(x+2*dx2) - func2(x-2*dx2)))/(28*dx2)
print(df11 - func1(x), df12 - func1(x))
print(df21 - 0.1 * func2(x), df22 - 0.1 * func2(x))
