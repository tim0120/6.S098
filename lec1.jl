using Pkg; Pkg.activate(".");
using Convex


# Define the variable
x = Variable()

# Only one of these is convex via DCP rules!
f2 = sqrt(1 + x*x)
f2 = sqrt(1 + square(x))
f2 = norm([1; x], 2)


## Example 2
y = Variable()
# This does not work!
num = (x - y)*(x - y)
num = square(x - y)
denom = 1 - max(x, y)
f = num / denom

# Have to use the quad over lin operator
f = quadoverlin(x - y, 1 - max(x, y))

# Let's solve a problem
using SCS
f2 = norm([1; x], 2)
prob = minimize(f2)
solve!(prob, SCS.Optimizer())
x.value

# We can add constraints
con = [
    x >= 1, 
    x <= 2
]
prob2 = minimize(f2, con)
solve!(prob2, SCS.Optimizer())
x.value

# We can add more constraints
push!(con, x ≥ 1.5)
prob2.constraints += x ≥ 2

# Solve again
solve!(prob2, SCS.Optimizer())
x.value

# We can also build up the objective
obj = 0
for i in 1:10
    obj += i*x
end
obj