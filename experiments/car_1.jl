using LinearAlgebra: norm
using Plots;

# z defined as z = [x_{0:T},u(0:T)]

T = 100 # Number of states after initial state 
Δt = 0.1 # Time step

x_0 = [0.0,0.0,0.0,0.0] # Initial state
x_f = [0,5.0,pi,0.0] # Final state
ns = length(x_0) # Number of states
nu = 2 # Number of controls

z_0 = [ones(ns*T);ones(nu*T)] # Initial guess
# λ_0 = rand(ns + ns*T + ns) # Initial guess for Lagrange multipliers
λ_0 = ones(ns*T + ns) # Initial guess for Lagrange multipliers

# Objective function 
# function eval_f(z)
#     return norm(x_f - z[ns*(T) + 1:ns*(T) + ns]) + norm(z[ns*(T+1) + 1:end])
# end
function eval_f(z)
    # println("Cost function controls: ", z[ns*(T) + 1:end])
    u = z[ns*T + 1:end]
    # display(u)
    # return norm(z[ns*(T) + 1:end])^2
    return sum(u.^2)
end

# Equality constraints
function eval_c(z)
    c = Vector{eltype(z)}(undef,ns*(T) + ns)
    x = Vector{eltype(z)}(undef,ns*T)
    u = Vector{eltype(z)}(undef,nu*T)

    # Breakout variables
    x = z[1:ns*T]
    u = z[ns*T + 1:end]

    # Dynamics constraints
    c[1] = x[1] - (x_0[1] + x_0[4]*cos(x_0[3])*Δt)
    c[2] = x[2] - (x_0[2] + x_0[4]*sin(x_0[3])*Δt)
    c[3] = x[3] - (x_0[3] + u[1]*Δt)
    c[4] = x[4] - (x_0[4] + u[2]*Δt)
    for i ∈ 2:T
        c[ns*(i-1) + 1] = x[ns*(i-1) + 1] - (x[ns*(i-2) + 1] + x[ns*(i-2) + 4]*cos(x[ns*(i-2) + 3])*Δt)
        c[ns*(i-1) + 2] = x[ns*(i-1) + 2] - (x[ns*(i-2) + 2] + x[ns*(i-2) + 4]*sin(x[ns*(i-2) + 3])*Δt)
        c[ns*(i-1) + 3] = x[ns*(i-1) + 3] - (x[ns*(i-2) + 3] + u[nu*(i-1) + 1]*Δt)
        c[ns*(i-1) + 4] = x[ns*(i-1) + 4] - (x[ns*(i-2) + 4] + u[nu*(i-1) + 2]*Δt)
    end

    # Final state constraint
    c[ns*(T) + 1:end] = x[ns*(T-1) + 1:end] - x_f

    # display(c)
    return(c)
end

# Solve
z_star = sqp_solve(eval_f,eval_c,z_0,λ_0);

# println("z_star = ",z_star)

# Wrap and round angles
z_star[3:ns:ns*T] = round.(z_star[3:ns:ns*T], digits = 1).%(2*pi)
# z_star[ns*(T) + 1:nu:end] = round.(z_star[ns*(T) + 1:nu:end], digits = 1).%(2*pi)

z_traj = [x_0;z_star[1:ns*(T)]]
x_star = z_star[ns*(T-1) + 1:ns*(T-1) + ns]
u_star = z_star[ns*(T) + 1:end]

# Print final state
# println("x_star = ", x_star)
# println("u_star = ", u_star)
println("f_0 = ", eval_f(z_0))
println("f_star = ", eval_f(z_star))

# Plot state trajectory
t = 0:Δt:T*Δt
p1 = plot(t,[z_traj[1:ns:end],z_traj[2:ns:end],z_traj[3:ns:end],z_traj[4:ns:end]],
     ylabel = ["x" "y" "θ" "v"], xlims = [t[1],t[end]], legend = false , layout = (4,1))

# Plot control trajectory
p2 = bar(t[1:end-1].+Δt/2,[u_star[1:nu:end],u_star[2:nu:end]],
     xlabel = ["" "time [s]"], ylabel = ["u1" "u2"], 
     xlims = [t[1],t[end]], legend = false , layout = (2,1))

plt_vc = plot(p1,p2,layout=(2,1),size = (750,1000))

display(plt_vc)

# Plot xy trajectory
plt_traj = plot(z_traj[1:ns:end],z_traj[2:ns:end],xlabel = "x", ylabel = "y", legend = false)
display(plt_traj)