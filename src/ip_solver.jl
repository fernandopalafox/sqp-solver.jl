# using ReverseDiff: GradientTape, JacobianTape, HessianTape, compile, gradient!, jacobian!, hessian!
# using LinearAlgebra: norm, diagm, I

# Interior-point solver 
# Based on Nocedal and Wright, 2nd ed., p. 564-569

function ip_solve(eval_f,eval_c_e,eval_c_i,x_0)

# println("step 1")
# Number of slack variables
length_x = length(x_0)
length_y = length(eval_c_e(x_0))
length_z = length(eval_c_i(x_0))
length_s = length_z

# println("step 2")
# Error function
function compute_E(x,s,y,z,mu,grad_f,A_e,A_i,S)    
    return maximum([
        norm(grad_f - A_e'*y - A_i'*z), 
        norm(S*z - mu.*ones(length_s)), 
        norm(eval_c_e(x)),
        norm(eval_c_i(x) - s)])
end

# println("step 3")
# Fraction to the boundary rule 
function find_a(s, z, p_s, p_z; tau = 0.995)
    # Range of alphas
    da = 0.001
    a_range = da:da:1.0

    # println("test ", all(s + a_range[1]*p_s .> (1 - tau)*s))
    # println("test ", findall(a -> all(s + a*p_s .>= (1 - tau)*s), a_range))
    # println("test ", findall(a -> all(s + a*p_s .>= (1 - tau)*s), a_range)[end])
    # println("")
    # println("test ", all(z + a_range[1]*p_z .>= (1 - tau)*z))
    # println("test ", findall(a -> all(z + a*p_z .>= (1 - tau)*z), a_range))
    # println("test ", findall(a -> all(z + a*p_z .>= (1 - tau)*z), a_range)[end])

    # Find alphas
    a_s = a_range[findall(a -> all(s + a*p_s .>= (1 - tau)*s), a_range)[end]]
    a_z = a_range[findall(a -> all(z + a*p_z .>= (1 - tau)*z), a_range)[end]]

    # Find alphas
    # a_s = a_range[argmax(i -> a_range[i], findall(a -> all(s + a*p_s .>= (1 - tau)*s), a_range))]
    # a_z = a_range[argmax(i -> a_range[i], findall(a -> all(z + a*p_z .>= (1 - tau)*z), a_range))]

    # Print whether inequalities are met or not

    # a_s = 0.001
    # a_z = 0.001

    return a_s, a_z
end

# println("step 4")
# Barrier parameters update
# To-do: Make this fancier
function update_barrier(mu_k)
    sigma = 0.2
    mu_k = sigma*mu_k
    return mu_k
end

# println("step 5")
# Lagrangian 
function eval_L(v)
    x = v[1:length_x]
    s = v[length_x + 1:length_x + length_s]
    y = v[length_x + length_s + 1: length_x + length_s + length_y]
    z = v[length_x + length_s + length_y + 1:end]
    return eval_f(x) - y'*eval_c_e(x) - z'*(eval_c_i(x) - s)
end

# println("step 6")
# Preallocate
grad_f       = rand(eltype(x_0),length_x)
A_e          = rand(eltype(x_0),length_y, length_x)
A_i          = rand(eltype(x_0),length_z, length_x)
hessian_L    = rand(eltype(x_0),length_x + length_s + length_y + length_z,length_x + length_s + length_y + length_z)
hessian_L_xx = rand(eltype(x_0),length_x,length_x)

x_k          = x_0
y_k          = ones(length_y)
z_k          = ones(length_z)

mu_k         = 100.0
s_k          = ones(eltype(x_0),length_s)

p_x          = rand(eltype(x_0),length_x)
p_s          = rand(eltype(x_0),length_s)
p_y          = rand(eltype(x_0),length_y)
p_z          = rand(eltype(x_0),length_z)

# println("step 7")
# Setup tapes 
tape_f        = GradientTape(eval_f,x_0)
tape_c_e      = JacobianTape(eval_c_e,x_0)
tape_c_i      = JacobianTape(eval_c_i,x_0)
tape_L        = HessianTape(eval_L,[x_k;s_k;y_k;z_k])

tape_f_comp   = compile(tape_f)
tape_c_e_comp = compile(tape_c_e)
tape_c_i_comp = compile(tape_c_i)
tape_L_comp   = compile(tape_L)

# Repeat until a stopping test for nonlinear program is satisfied
for counter in 1:17
    # println("step 8")
    # Construct S and Z matrices
    S = diagm(s_k)
    Z = diagm(z_k)

    # Evaluate gradient of objective
    gradient!(grad_f,tape_f_comp,x_k)

    # Evaluate Jacobians
    jacobian!(A_e,tape_c_e_comp,x_k)
    jacobian!(A_i,tape_c_i_comp,x_k)

    # Evaluate Hessian of Lagrangian (keep values wrt to x only) 
    hessian!(hessian_L,tape_L_comp,[x_k;s_k;y_k;z_k])
    hessian_L_xx = hessian_L[1:length_x,1:length_x]

    # println("step 9")
    # Repeat until error function is leq mu_k
    counter_inner = 0
    while(compute_E(x_k,s_k,y_k,z_k,mu_k,grad_f,A_e,A_i,S) > mu_k)   

        # Solve primal-dual system to find steps     
        # println("step 10")    
         p = 
        [hessian_L_xx             zeros(length_x,length_z) -A_e'                    -A_i';
         zeros(length_z,length_x) Z                        zeros(length_z,length_y) S;
         A_e                      zeros(length_y,length_z) zeros(length_y,length_y) zeros(length_y,length_z)
         A_i                      -1.0*I(length_z)         zeros(length_z,length_y) zeros(length_z,length_z)]\
        -[grad_f - A_e'*y_k - A_i'*z_k;
          S*z_k - mu_k.*ones(length_s);
          eval_c_e(x_k);
          eval_c_i(x_k) - s_k]
         
        # println("step 11")
         # Breakout steps
         p_x = p[1:length_x]
         p_s = p[length_x + 1:length_x + length_s]
         p_y = p[length_x + length_s + 1: length_x + length_s + length_y]
         p_z = p[length_x + length_s + length_y + 1:end]
         # Print steps
        # println("   p_x = ", p_x, " p_s = ", p_s, " p_y = ", p_y, " p_z = ", p_z)

        # println("step 12")
        # Compute fraction to the boundary 
        a_s, a_z = find_a(s_k, z_k, p_s, p_z;  tau = 0.995)

        # Print steps
        # println("   a_s = ", a_s, " a_z = ", a_z)

        # Print previous and new
        # println("   previous S:", S)
        # println("   New S     :", diagm(s_k + a_s.*p_s))
        # println("   previous Z:", S)
        # println("   New Z     :", diagm(z_k + a_z.*p_z))

        # Update variables
        x_k = x_k + a_s.*p_x
        s_k = s_k + a_s.*p_s
        y_k = y_k + a_z.*p_y
        z_k = z_k + a_z.*p_z

        # Re-compute gradient, Jacobian, and Hessian
        S = diagm(s_k)
        Z = diagm(z_k)
        gradient!(grad_f,tape_f_comp,x_k)
        jacobian!(A_e,tape_c_e_comp,x_k)
        jacobian!(A_i,tape_c_i_comp,x_k)
        hessian!(hessian_L,tape_L_comp,[x_k;s_k;y_k;z_k])
        hessian_L_xx = hessian_L[1:length_x,1:length_x]

        # Update counter
        counter_inner += 1

        # Print new error values
        # println("inner iteration = ",counter_inner," mu_k = ",mu_k," E = ", compute_E(x_k,s_k,y_k,z_k,mu_k,grad_f,A_e,A_i,S))
    end
    # Update barrier parameter (Fiacco and McCormick strategy)
    mu_k = update_barrier(mu_k)
    # println("   outer iteration = ", counter, " inner iterations = ", counter_inner, " barrier parameter = ", mu_k)
end

# print("     mu_k = ", mu_k,"\n")

return x_k, [y_k; z_k]
end
