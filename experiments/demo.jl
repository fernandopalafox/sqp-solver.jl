# Run this to test code

# Objective function
# Excercise 18.3 in Nocedal
function eval_f(x)
    return exp(prod(x)) - 0.5*(x[1]^3 + x[2]^3 + 1)^2 
end

# Constraints
function eval_c(x)
    c = Vector{typeof(x[1])}(undef,3)
    c[1] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 - 10
    c[2] = x[2]*x[3] - 5*x[4]*x[5] 
    c[3] = x[1]^3 + x[2]^3 + 1 
    print(c)
    return all(c .== 0)
end

# Starting point
x_0 = [-1.71, 1.59, 1.82,-0.763,-0.763]

# Run problem 