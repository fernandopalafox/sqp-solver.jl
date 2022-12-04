using ReverseDiff: GradientTape, JacobianTape, HessianTape, compile, gradient!, jacobian!, hessian!
using LinearAlgebra: norm, rank, eigvals

function sqp_solve(eval_f,eval_c,x_0,λ_0)
    # Set initial iterates
    counter = 0
    x_k = x_0
    λ_k = λ_0 

    # Preallocate 
    length_x     = length(x_0)
    length_λ     = length(λ_0)
    grad_f       = similar(x_k)
    hessian_L    = rand(length_x + length_λ,length_x + length_λ)
    hessian_L_xx = rand(length_x,length_x)
    A            = rand(length_λ,length_x)
    zero_mat     = zeros(eltype(x_0),length_λ,length_λ)

    # Lagrangian
    function eval_L(v)
        x = v[1:length_x]
        λ = v[length_x + 1:end]
        return eval_f(x) - λ'*eval_c(x)
    end

    # Setup tapes 
    tape_f = GradientTape(eval_f,x_0)
    tape_c = JacobianTape(eval_c,x_0)
    tape_L = HessianTape(eval_L,rand(length_x + length_λ))
    tape_f_comp = compile(tape_f)
    tape_c_comp = compile(tape_c)
    tape_L_comp = compile(tape_L)

    ϕ = 100.0
    while ϕ > 1e-13

        # Evaluate gradient of objective function  
        gradient!(grad_f,tape_f_comp,x_k)

        # Evaluate Hessian of Lagrangian (only at x)
        hessian!(hessian_L,tape_L_comp,[x_k;λ_k])
        hessian_L_xx = hessian_L[1:length_x,1:length_x]

        # Evaluate constraints
        c = eval_c(x_k)

        # Evaluate contraint jacobian
        jacobian!(A,tape_c_comp,x_k)

        # Evaluate merit function 
        ϕ = norm([grad_f - A'*λ_k;c])^2

        # Solve Lagrangian QP 
        p_p = ([hessian_L_xx -A';A zero_mat])\[-grad_f; -c]
        x_k = x_k + p_p[1:length_x]
        λ_k = p_p[length_x+1:end]

        println("   ϕ_",counter," = ", ϕ)

        counter = counter + 1
    end
    return(x_k)
end