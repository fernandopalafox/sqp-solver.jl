using LinearAlgebra: norm
using Plots;
include("../src/sqp_solve.jl")

# z defined as z = [x_{0:T},u(0:T)]

T = 100 # Number of states after initial state 
Δt = 0.1 # Time step

x_0 = [0.0,0.0,0.0,0.0] # Initial state
x_f = [0,5.0,pi,0.0] # Final state
x_f = [1.0,0,pi,0.0] # Good example
# x_f = [1.0,1.0,pi/4,0.0]
# x_f = [3.0,0.0,3*pi/2,0.0]
ns = length(x_0) # Number of states
nu = 2 # Number of controls

z_0 = [rand(ns*T);rand(nu*T)] # Initial guess
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
z_star_0 = z_0
z_star_1 = sqp_solve(eval_f,eval_c,z_0,λ_0; counter_max = 3);
z_star_2 = sqp_solve(eval_f,eval_c,z_0,λ_0; counter_max = 4);
z_star_3 = sqp_solve(eval_f,eval_c,z_0,λ_0; counter_max = 5);
z_star_4 = sqp_solve(eval_f,eval_c,z_0,λ_0; counter_max = 6);
z_star_5 = sqp_solve(eval_f,eval_c,z_0,λ_0; counter_max = 7);
z_star_6 = sqp_solve(eval_f,eval_c,z_0,λ_0; counter_max = 8);
z_star_7 = sqp_solve(eval_f,eval_c,z_0,λ_0; counter_max = 9);

z_star_0[3:ns:ns*T] = round.(z_star_0[3:ns:ns*T], digits = 1).%(2*pi)
z_star_1[3:ns:ns*T] = round.(z_star_1[3:ns:ns*T], digits = 1).%(2*pi)
z_star_2[3:ns:ns*T] = round.(z_star_2[3:ns:ns*T], digits = 1).%(2*pi)
z_star_3[3:ns:ns*T] = round.(z_star_3[3:ns:ns*T], digits = 1).%(2*pi)
z_star_4[3:ns:ns*T] = round.(z_star_4[3:ns:ns*T], digits = 1).%(2*pi)
z_star_5[3:ns:ns*T] = round.(z_star_5[3:ns:ns*T], digits = 1).%(2*pi)
z_star_6[3:ns:ns*T] = round.(z_star_6[3:ns:ns*T], digits = 1).%(2*pi)
z_star_7[3:ns:ns*T] = round.(z_star_7[3:ns:ns*T], digits = 1).%(2*pi)

z_traj_0 = [x_0;z_star_0[1:ns*(T)]]
z_traj_1 = [x_0;z_star_1[1:ns*(T)]]
z_traj_2 = [x_0;z_star_2[1:ns*(T)]]
z_traj_3 = [x_0;z_star_3[1:ns*(T)]]
z_traj_4 = [x_0;z_star_4[1:ns*(T)]]
z_traj_5 = [x_0;z_star_5[1:ns*(T)]]
z_traj_6 = [x_0;z_star_6[1:ns*(T)]]
z_traj_7 = [x_0;z_star_7[1:ns*(T)]]

# # Plot state trajectory
t = 0:Δt:T*Δt

color_grad = cgrad(:acton, rev = true)

# Plot xy trajectory
p_xy = plot(
    [z_traj_3[1:ns:end],z_traj_4[1:ns:end],z_traj_5[1:ns:end],z_traj_6[1:ns:end],z_traj_7[1:ns:end]],
    [z_traj_3[2:ns:end],z_traj_4[2:ns:end],z_traj_5[2:ns:end],z_traj_6[2:ns:end],z_traj_7[2:ns:end]],
    line = :solid,
    color= cgrad(:copper, rev = true),
    rev = true,
    line_z = (0:5)',
    lw = 3,
    primary = false,
    xticks = false, yticks = false,
    grid = true,
    showaxis = false,
    xtickfont=font("Computer Modern", 12), 
    ytickfont=font("Computer Modern",12), 
    xguidefont=font("Computer Modern",15),
    yguidefont=font("Computer Modern",15), 
    legendfont=font("Computer Modern",12),
    legend = false,
    size = (350,350))
display(p_xy)

savefig(p_xy, "figures/visual.png")