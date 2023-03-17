def u(x):
    return x**2

a = 1.0
dx = 1e-1

# Forwards difference
dudx = (u(a+dx) - u(a))/(dx)

print('Forwards diff dudx = ', dudx)

# Backwards difference

dudx = (u(a) - u(a-dx))/(dx)

print('Backwards diff dudx = ', dudx)

# Central difference

dudx = (u(a+dx) - u(a-dx))/(2*dx)

print('Central diff dudx = ', dudx)

# second order central difference

dudx = (u(a+dx) - 2*u(a) + u(a-dx))/(dx**2)

print('Second order central diff dudx = ', dudx)
