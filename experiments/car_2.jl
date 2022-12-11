using LinearAlgebra: norm
using Plots;
include("../src/sqp_solve_ineq.jl")

# z defined as z = [x_{0:T},u(0:T)]

T = 7 # Number of states after initial state 
Δt = 1 # Time step

x_0 = [0.0,0.0,0.0,0.0] # Initial state
x_f = [0,5.0,pi,0.0] # Final state
ns = length(x_0) # Number of states
nu = 2 # Number of controls

z_0 = [ones(ns*T);ones(nu*T)] # Initial guess
z_o = [zeros(ns*T);zeros(nu*T)]
λ_0 = ones(ns*T + ns + T) # Initial guess for Lagrange multipliers

# Objective function 
function eval_f(z)
    u = Vector{eltype(z)}(undef,nu*T)
    u = z[ns*T + 1:end]
    return sum(u.^2)
end

# Equality constraints
function eval_c_e(z)
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

# Inequality Constraints
function eval_c_i(z)
    x = Vector{eltype(z)}(undef,ns*T)

    # Breakout variables
    x = z[1:ns*T]

    # Wall constraint
    c = -x[1:ns:ns*T] .+ 1.0

    return(c)
end


# Solve
z_star = sqp_solve_ineq(eval_f,eval_c_e,eval_c_i,z_0,λ_0)

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

## 

# Plot state trajectory
t = 0:Δt:T*Δt
# p1 = plot(t,[z_traj[1:ns:end],z_traj[2:ns:end],z_traj[3:ns:end],z_traj[4:ns:end]],
#      ylabel = ["x" "y" "θ" "v"], xlims = [t[1],t[end]], legend = false , layout = (4,1))

# # Plot control trajectory
# p2 = bar(t[1:end-1].+Δt/2,[u_star[1:nu:end],u_star[2:nu:end]],
#      xlabel = ["" "time [s]"], ylabel = ["u1" "u2"], 
#      xlims = [t[1],t[end]], legend = false , layout = (2,1))

# plt_vc = plot(p1,p2,layout=(2,1, heights = [2/5, 2/5, 2/5, 2/5, 2/5]),size = (750,750))

y_ticks_px = round.(minimum(z_traj[1:ns:end]):(maximum(z_traj[1:ns:end]) - minimum(z_traj[1:ns:end]))/2:maximum(z_traj[1:ns:end]),digits = 1)
y_ticks_py = round.(minimum(z_traj[2:ns:end]):(maximum(z_traj[2:ns:end]) - minimum(z_traj[2:ns:end]))/2:maximum(z_traj[2:ns:end]),digits = 1)
y_ticks_pt = round.(minimum(z_traj[3:ns:end]):(maximum(z_traj[3:ns:end]) - minimum(z_traj[3:ns:end]))/2:maximum(z_traj[3:ns:end]),digits = 1)
y_ticks_pv = round.(minimum(z_traj[4:ns:end]):(maximum(z_traj[4:ns:end]) - minimum(z_traj[4:ns:end]))/2:maximum(z_traj[4:ns:end])+0.6,digits = 1)
y_ticks_pu1 = [0.0,0.4,0.7]
y_ticks_pu2 = [-0.6,0.0,0.6]

px = plot(t,z_traj[1:ns:end], xlims = [t[1],t[end]],label = "x",lw = 3, yticks = y_ticks_px,color = 2)
py = plot(t,z_traj[2:ns:end], xlims = [t[1],t[end]],label = "y",lw = 3, yticks = y_ticks_py,color = 2)
pt = plot(t,z_traj[3:ns:end], xlims = [t[1],t[end]],label = "Θ",lw = 3, yticks = y_ticks_pt,color = 2)
pv = plot(t,z_traj[4:ns:end], xlims = [t[1],t[end]],label = "v",lw = 3, yticks = y_ticks_pv, ylim = [y_ticks_pv[1], y_ticks_pv[3]+0.1],color = 2)
pu1 = bar(t[1:end-1].+Δt/2,u_star[1:nu:end], xlims = [t[1],t[end]], ylim = [y_ticks_pu1[1], y_ticks_pu1[3]], yticks = y_ticks_pu1, label = "u1",color = 2)
pu2 = bar(t[1:end-1].+Δt/2,u_star[2:nu:end], xlims = [t[1],t[end]], yticks = y_ticks_pu2, ylim = [y_ticks_pu2[1], y_ticks_pu1[3]-0.1], label = "u2", xlabel = "time [s]",color = 2)
p_z = plot(px,py,pt,pv,pu1,pu2,layout=(6,1),size = (0.8*750,750),
xtickfont=font("Computer Modern", 12), 
ytickfont=font("Computer Modern",12), 
xguidefont=font("Computer Modern",15), 
legendfont=font("Computer Modern",12),
fmt = :png)
display(p_z,)

savefig(p_z, "figures/car_2_z.png")

# Plot xy trajectory
p_xy = plot([z_traj_1[1:ns:end],z_traj[1:ns:end]],[z_traj_1[2:ns:end],z_traj[2:ns:end]],xlabel = "x", ylabel = "y", labels = ["P1" "P2"],
lw = 3,
xlim = [0,1.5],
xtickfont=font("Computer Modern", 12), 
ytickfont=font("Computer Modern",12), 
xguidefont=font("Computer Modern",15),
yguidefont=font("Computer Modern",15), 
legendfont=font("Computer Modern",12),
size = (350,350))
display(p_xy)

savefig(p_xy, "figures/car_2_xy.png")