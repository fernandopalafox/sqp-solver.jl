using ReverseDiff: GradientTape, JacobianTape, HessianTape, compile, gradient!, jacobian!, hessian!
using LinearAlgebra: norm, rank, eigvals, diagm, I 
include("../src/ip_solver.jl")

function sqp_solve_ineq(eval_f,eval_c_e,eval_c_i,x_0,λ_0; ϕ_min = 0.05)
    # Set initial iterates
    counter = 0
    x_k = x_0
    λ_k = λ_0 

    # Preallocate 
    length_x     = length(x_0)
    length_λ     = length(λ_0)
    length_y    = length(eval_c_e(x_0))
    length_z    = length(eval_c_i(x_0))
    @assert length_λ == length_y + length_z "length of λ_0 must be equal to length of y + length of z"
    grad_f       = similar(x_k)
    hessian_L    = rand(length_x + length_λ,length_x + length_λ)
    hessian_L_xx = rand(length_x,length_x)
    A_e          = rand(length_y,length_x) 
    A_i          = rand(length_z,length_x) 
    zero_mat     = zeros(eltype(x_0),length_λ,length_λ)

    # Lagrangian
    function eval_L(v)
        x = v[1:length_x]
        y = v[length_x + 1: length_x + length_y]
        z = v[length_x + length_y + 1:end]
        return eval_f(x) - y'*eval_c_e(x) - z'*eval_c_i(x)
    end

    # Setup tapes 
    tape_f = GradientTape(eval_f,x_0)
    tape_c_e = JacobianTape(eval_c_e,x_0)
    tape_c_i = JacobianTape(eval_c_i,x_0)
    tape_L = HessianTape(eval_L,rand(length_x + length_λ))
    tape_f_comp = compile(tape_f)
    tape_c_e_comp = compile(tape_c_e)
    tape_c_i_comp = compile(tape_c_i)
    tape_L_comp = compile(tape_L)

    ϕ = 100.0
    while ϕ > ϕ_min

        # Evaluate gradient of objective function  
        gradient!(grad_f,tape_f_comp,x_k)

        # Evaluate Hessian of Lagrangian (only at x)
        hessian!(hessian_L,tape_L_comp,[x_k;λ_k])
        hessian_L_xx = hessian_L[1:length_x,1:length_x]

        # Evaluate constraints at x_k
        c_e = eval_c_e(x_k)
        c_i = eval_c_i(x_k)

        # Evaluate objective at x_k
        f_k = eval_f(x_k)

        # Evaluate contraint jacobians
        jacobian!(A_e,tape_c_e_comp,x_k)
        jacobian!(A_i,tape_c_i_comp,x_k)

        # Evaluate merit function (norm square of KKT conditions)
        # To-do: Add a way of accounting for inequality constraints in the merit function
        ϕ = norm([grad_f - A_e'*λ_k[1:length_y] - A_i'*λ_k[length_y+1:end];c_e])^2
        println("ϕ_",counter," = ", ϕ)
        # println("   grad_f ", norm(grad_f))
        # println("   A_e'*λ_k ", norm(A_e'*λ_k[1:length_y]))
        # println("   A_i'*λ_k ", norm(A_i'*λ_k[length_y+1:end]))
        # println("   grad_f - A_e'*λ_k - A_i'*λ_k ", norm(grad_f - A_e'*λ_k[1:length_y] - A_i'*λ_k[length_y+1:end]))
        # println("   c_e ", norm(c_e))
        if ϕ > 1000
            println("ϕ is too large, exiting...")
            break
        end

        # Solve Lagrangian QP using interior point method
        # println("       Starting interior point method...")
        # println("       eigenvalues of hessian", eigvals(hessian_L_xx))
        p_k, λ_kp1 = 
        ip_solve(p -> 1/2*p'*hessian_L_xx*p + grad_f'*p + f_k,
                 p -> A_e*p + c_e,
                 p -> A_i*p + c_i,
                 zeros(length_x))
        
        # Update iterates
        x_k = x_k + 0.05*p_k
        λ_k = λ_kp1

        # Test if it's a good step or not, reduce step otherwise
        # if norm([grad_f - A_e'*λ_k[1:length_y] - A_i'*λ_k[length_y+1:end];c_e])^2 > > phi

        # Update counter
        counter = counter + 1
    end
    return(x_k)
end