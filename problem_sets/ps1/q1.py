import numpy as np

func1 = np.exp
func2 = lambda x : np.exp(0.1 * x)
epsilon = np.power(2.0,-53)
dx1 = np.power(6 * epsilon, 1.0/5)
dx2 = np.power(6 * epsilon, 1.0/4) #slightly different values of dx to show that the first one is correct
dx3 = np.power(6 * epsilon, 1.0/6)
x = 1
df11 = (8*(func1(x+dx1) - func1(x-dx1)) - (func1(x+2*dx1) - func1(x-2*dx1)))/(12*dx1)
df21 = (8*(func2(x+dx1) - func2(x-dx1)) - (func2(x+2*dx1) - func2(x-2*dx1)))/(12*dx1)
df12 = (8*(func1(x+dx2) - func1(x-dx2)) - (func1(x+2*dx2) - func1(x-2*dx2)))/(12*dx2)
df22 = (8*(func2(x+dx2) - func2(x-dx2)) - (func2(x+2*dx2) - func2(x-2*dx2)))/(12*dx2)
df13 = (8*(func1(x+dx3) - func1(x-dx3)) - (func1(x+2*dx3) - func1(x-2*dx3)))/(12*dx3)
df23 = (8*(func2(x+dx3) - func2(x-dx3)) - (func2(x+2*dx3) - func2(x-2*dx3)))/(12*dx3)
print(df11 - func1(x), df12 - func1(x), df13 - func1(x))
print(df21 - 0.1 * func2(x), df22 - 0.1 * func2(x), df23 - 0.1 * func2(x))
