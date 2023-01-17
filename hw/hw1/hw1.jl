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
T = 30; Tstart = 15; Tend = 20; Smin = 25; Smax = 35; L = 3.7;
# for fun
# T = 30; Tstart = 5; Tend = 25; Smin = 25; Smax = 35; L = 3.7;

x = Variable(T + 1); y = Variable(T + 1);
p = [x y]
size(p) # sanity check

dists = p[2:T+1, :] - p[1:T, :]
normVec = [norm(dists[i, :], 2) for i in 1:T]

consts = [
  x[1] == 0, # initial condition on x
  x[1:T] <= x[2:T+1],
  y[1:Tstart] == 0,
  y[Tend:T+1] == L,
  y >= 0,
  y <= L,
]

for i in 1:T
  push!(consts, normVec[i] <= Smax)
  if i in 1:Tstart-2 || i in Tend-1:T
    push!(consts, x[i+1] - x[i] >= Smin)
  end
end

acc = p[3:T+1, :] - 2*p[2:T, :] + p[1:T-1, :]
size(acc)
obj = sum([square(norm(acc[i, :])) for i in 1:T-1])

problem = minimize(obj, consts)
solve!(problem, SCS.Optimizer())

x
y

using Plots
# plotting position of car
plot(x.value, y.value)
plot!(legend=false)
title!("Lane Change in Position Space")
xlabel!("X Position (m)")
ylabel!("Y Position (m)")

# plotting speed vs. time
xs = x.value
ys = y.value
delta = hcat(
  xs[2:T+1] - xs[1:T],
  ys[2:T+1] - ys[1:T]
)
speeds = [norm(delta[i, :], 2) for i in 1:T]
plot(1:T, speeds)
plot!(legend=false)
title!("Speed vs. Time During Lane Change")
xlabel!("Time (s)")
ylabel!("Speed (m/s)")