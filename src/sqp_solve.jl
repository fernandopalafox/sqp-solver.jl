using ReverseDiff: GradientTape, JacobianTape, HessianTape, compile, gradient!, jacobian!, hessian!
using LinearAlgebra: norm

function sqp_solve()
    # Set initial iterates
    x_k = x_0
    λ_k = λ_0 
    println("\nx_0 = ", x_0)

    # Preallocate 
    length_x     = length(x_0)
    length_λ     = length(λ_0)
    grad_f       = similar(x_k)
    hessian_L    = rand(length_x + length_λ,length_x + length_λ)
    hessian_L_xx = rand(length_x,length_x)
    A            = rand(length_λ,length_x)
    zero_mat     = zeros(typeof(x_0[1]),length_λ,length_λ)

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

    for i ∈ 1:100
        # Evaluate gradient of objective function  
        gradient!(grad_f,tape_f_comp,x_k)

        # Evaluate Hessian of Lagrangian (only at x)
        hessian!(hessian_L,tape_L_comp,[x_k;λ_k])
        hessian_L_xx = hessian_L[1:length_x,1:length_x]

        # Evaluate constraints
        c = eval_c(x_k)

        # Evaluate contraint jacobian
        jacobian!(A,tape_c_comp,x_k)

        # Solve for Newton step 
        p = inv([hessian_L_xx -A';A zero_mat])*[-grad_f + A'*λ_k; -c]
        p_x_k = p[1:length_x]
        p_λ_k = p[length_x+1:end]

        # Set new iterates
        x_k = x_k + p_x_k
        λ_k = λ_k + p_λ_k

        # Check merit function 
        # phi = norm([grad_f - A'*λ_k;eval_c(x_k)])

        # Print
        println("   p_x_",i," = ", p_x_k)
        println("   p_λ_",i," = ", p_λ_k)
        println("x_",i," = ", x_k)
    end
    return(x_k)
end