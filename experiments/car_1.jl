using LinearAlgebra: norm

# z defined as z = [x_{0:T},u(0:T)]

T = 200 # Number of states after initial state 
Δt = 0.01 # Time step
V = 1 # constant velocity

x_0 = [0.1,0.2,0.3] # Initial state
x_f = [10,10,0] # Final state

z_0 = [x_0;rand(3*T);rand(T)] # Initial guess
λ_0 = rand(3*T + 3) # Initial guess for Lagrange multipliers

# Objective function 
function eval_f(z)
    return norm(x_f - z[3*T+1:3*T+3]) + sum(z[3*T+3:end].^2)
end

# Equality constraints
function eval_c(z)
    c = Vector{eltype(z)}(undef,3*(T)+3)
    c[1] = z[1] - x_0[1]
    c[2] = z[2] - x_0[2]
    c[3] = z[3] - x_0[3]
    for i ∈ 1:T
        c[3*(i) + 1] = z[3*(i) + 1] - V*cos(z[3*(i)])*Δt
        c[3*(i) + 2] = z[3*(i) + 2] - V*sin(z[3*(i)])*Δt
        c[3*(i) + 3] = z[3*(i) + 3] - z[3*(T+1) + i]*Δt
    end
    return(c)
end

# Solve
z_star = sqp_solve(eval_f,eval_c,z_0,λ_0);

# Print final state
println("x_star = ", z_star[3*T+1:3*T+3])