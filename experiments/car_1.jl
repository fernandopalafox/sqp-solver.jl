using LinearAlgebra: norm
using Plots

# z defined as z = [x_{0:T},u(0:T)]

T = 2 # Number of states after initial state 
Δt = 0.1 # Time step

x_0 = [0.0,0.0,0.0,0.0] # Initial state
x_f = [5.0,0.0,0.0,0.0] # Final state
ns = length(x_0) # Number of states
nu = 2 # Number of controls

z_0 = [rand(ns*T);rand(nu*T)] # Initial guess
# λ_0 = rand(ns + ns*T + ns) # Initial guess for Lagrange multipliers
λ_0 = rand(ns*T + ns) # Initial guess for Lagrange multipliers

# Objective function 
# function eval_f(z)
#     return norm(x_f - z[ns*(T) + 1:ns*(T) + ns]) + norm(z[ns*(T+1) + 1:end])
# end
function eval_f(z)
    # println("Cost function controls: ", z[ns*(T) + 1:end])
    return norm(z[ns*(T) + 1:end])
end

# Equality constraints
function eval_c(z)
    c = Vector{eltype(z)}(undef,ns*(T) + ns)

    # Dynamics constraint
    c[1] = z[1] - x_0[4]*cos(x_0[3])*Δt
    c[2] = z[2] - x_0[4]*sin(x_0[3])*Δt
    c[3] = z[3] - z[ns*(T) + 1]*Δt
    c[4] = z[4] - z[ns*(T) + 2]*Δt
    for i ∈ 2:T
        c[ns*(i-1) + 1] = z[ns*(i-1) + 1] - z[ns*(i-2) + 4]*cos(z[ns*(i-2) + 3])*Δt
        c[ns*(i-1) + 2] = z[ns*(i-1) + 2] - z[ns*(i-2) + 4]*sin(z[ns*(i-2) + 3])*Δt
        c[ns*(i-1) + 3] = z[ns*(i-1) + 3] - z[ns*(T) + nu*(i-1) + 1]*Δt
        c[ns*(i-1) + 4] = z[ns*(i-1) + 4] - z[ns*(T) + nu*(i-1) + 2]*Δt

        # println("control t = ",z[ns*(T) + nu*(i-1) + 1])
        # println("control v = ",z[ns*(T) + nu*(i-1) + 2])
    end

    # Final state constraint
    # println("final state = ",z[ns*(T-1) + 1:ns*(T-1) + ns])
    c[ns*(T) + 1:ns*(T) + ns] = z[ns*(T-1) + 1:ns*(T-1) + ns] - x_f 

    # display(c)
    return(c)
end

# Solve
z_star = sqp_solve(eval_f,eval_c,z_0,λ_0);

# Wrap angles
z_star[3:ns:ns*T] = z_star[3:ns:ns*T].%(2*pi)
z_star[ns*(T) + 1:nu:end] = z_star[ns*(T) + 1:nu:end].%(2*pi)

z_traj = [x_0;z_star[1:ns*(T)]]
x_star = z_star[ns*(T-1) + 1:ns*(T-1) + ns]
u_star = z_star[ns*(T) + 1:end]

# Print final state
println("x_star = ", x_star)
println("u_star = ", u_star)
println("f_0 = ", eval_f(z_0))
println("f_star = ", eval_f(z_star))

# Plot state trajectory
t = 0:Δt:T*Δt
plot(t,[z_traj[1:ns:end],z_traj[2:ns:end],z_traj[3:ns:end],z_traj[4:ns:end]],xlabel = ["" "" "" "time [s]"], ylabel = ["x" "y" "θ" "v"], legend = false , layout = (4,1))