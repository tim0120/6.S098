using Pkg; Pkg.activate(".");
using Convex, SCS

# 3.33
x = Variable(); y = Variable(Positive())
sq = sqrt(y)
qol = quadoverlin(x, sq)
f = norm([1; qol])

# 4.3a
x = Variable(); y = Variable()
cons = sum([invpos(x) invpos(y)])
consts = [cons <= 1; x >= 0; y >= 0]
f = norm([x; y], 2)
problem = minimize(f, consts)
solve!(problem, SCS.Optimizer())

# 4.3b
x = Variable(); y = Variable()
cons = square(geomean(x, y))
consts = [cons >= 1, x >= 0, y >= 0]
problem = minimize(f, consts)
solve!(problem, SCS.Optimizer())

# 4.3c
x = Variable(); y = Variable()
left = quadoverlin(x + y, sqrt(y))
right = x - y + 5
consts = [left <= right]
problem = minimize(f, consts)
solve!(problem, SCS.Optimizer())

# 4.3d
x = Variable(); y = Variable(); z = Variable()
left = x + z
right = sqrt(square(geomean(x, y)) - square(z))
consts = [left <= right]
problem = minimize(f, consts)
solve!(problem, SCS.Optimizer())

# 16.6
T = 30; Tstart = 15; Tend = 20; Smin = 25; Smax = 35; L = 3.7
