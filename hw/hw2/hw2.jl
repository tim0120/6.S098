using Pkg; Pkg.activate(".");
Pkg.add("ECOS")
using Convex, SCS, ECOS

# 20.14
C_max = 3
N = 4
T = 90
q_init = vec([20. 0. 30. 25.])
gamma = [0.5 0.3 2 0.6]
target_mins = zeros(T+1, N)
target_charge = [60 100 75 125]
for t in 1:T+1
  factor = reshape((t / (T+1)).^gamma, size(target_charge))
  target_mins[t, :] = factor .* target_charge
end

c = Variable(T, N)
q::AbstractArray{Union{Convex.AdditionAtom, Float64}} = zeros(T+1, N)
s::AbstractArray{Union{Convex.MaxAtom, Float64}} = zeros(T+1, N)

q[1, :] = q_init
for t in 1:T
  for i in 1:N
    q[t+1, i] = q[t, i] + c[t, i]
  end
end
s = max.(0, target_mins - q)

cons = [c >= 0]
for t in 1:T
  cons += sum(c[t, :]) <= C_max
end

obj = sumsquares(vcat(s...)) / (N * (T+1))
problem = minimize(obj, cons)
solve!(problem, ECOS.Optimizer())

Pkg.add("Plots")
using Plots
areaplot(1:T, c.value)


# 16.8
T = 12; h=0.1;
n = Int(T/h)

function trajectory(prev::Bool, x_prev::Array{Float64})
  x = Variable(n+1, 2)
  v = Variable(n+1, 2)
  a = Variable(n+1, 2)
  cons = [
    x[1, :] == [-5 0],
    x[n+1, :] == [6 1],
    v[1, :] == [2 0],
    v[n+1, :] == [0 0]
  ]
  for i in 1:n
    cons += v[i+1, :] == v[i, :] + h*a[i, :]
    cons += x[i+1, :] == x[i, :] + h/2*(v[i+1, :] + v[i, :])
  end
  
  if prev == true
    for i in 2:n
      cons += dot(x_prev[i, :], x[i, :]) >= norm(x_prev[i, :], 2)
    end
  end

  obj = h * sum([norm(a[i, :], 2) for i in 1:n+1])
  problem = minimize(obj, cons)
  solve!(problem, SCS.Optimizer())

  return x.value, v.value, a.value
end

x_no_roundabout, v_nr, a_nr = trajectory(false, [0.]);
x_1iter, v_1, a_1 = trajectory(true, x_no_roundabout);

x_prev = x_1iter
for i in 1:500
  x_prev, v_p, a_p = trajectory(true, x_prev)
end

plot(x_no_roundabout[:, 1], x_no_roundabout[:, 2], label="No Roundabout")
plot!(x_1iter[:, 1], x_1iter[:, 2], label="1 Correction Iteration")
plot!(x_prev[:, 1], x_prev[:, 2], label="Converged Trajectory")
plot!(title="Bike Trajectories")
angles = range(0, 2pi, 100);
plot!(cos.(angles), sin.(angles), label="")
plot!(aspect_ratio=:equal)

plot([norm(v_nr[i, :], 2) for i in 1:n+1], label="Speed (No Roundabout)")
plot!([norm(a_nr[i, :], 2) for i in 1:n+1], label="Acceleration (No Roundabout)")
plot!([norm(v_1[i, :], 2) for i in 1:n+1], label="Speed (No Roundabout)")
plot!([norm(a_1[i, :], 2) for i in 1:n+1], label="Acceleration (No Roundabout)")
plot!(title="Speed/Acceleration vs. Time", xlabel="Time Steps", ylabel="Speed")