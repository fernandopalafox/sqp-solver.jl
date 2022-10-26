# Excercise 18.3 in Nocedal

# Objective function
function eval_f(x)
    return exp(prod(x)) - 0.5*(x[1]^3 + x[2]^3 + 1)^2 
end

# Constraints
function eval_c(x)
    c = Vector{typeof(x[1])}(undef,3)
    c[1] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 - 10
    c[2] = x[2]*x[3] - 5*x[4]*x[5] 
    c[3] = x[1]^3 + x[2]^3 + 1 
    return c
end

# Starting point
x_0 = [-1.71, 1.59, 1.82,-0.763,-0.763]
λ_0 = [1.0,2.0,3.0]

# Run problem 
x_star = sqp_solve()

# Solution 
x_star_actual = [-1.8, 1.7, 1.9,-0.8,-0.8];

# # Analytical gradient
# function eval_grad_f(x)
#     ∇f = Vector{typeof(x[1])}(undef,5)
#     ∇f[1] = x[2]*x[3]*x[4]*x[5]*exp(prod(x)) - 3*(x[1]^3 + x[2]^3 + 1)*x[1]^2
#     ∇f[2] = x[1]*x[3]*x[4]*x[5]*exp(prod(x)) - 3*(x[1]^3 + x[2]^3 + 1)*x[2]^2
#     ∇f[3] = x[1]*x[2]*x[4]*x[5]*exp(prod(x))
#     ∇f[4] = x[1]*x[2]*x[3]*x[5]*exp(prod(x))
#     ∇f[5] = x[1]*x[2]*x[3]*x[4]*exp(prod(x))

#     return ∇f
# end