# Generate Dubins car trajectory given a control sequence and initial state

V = 1.0
x_0 = [0.0,0.0,0.0] # Initial state
u = [1.0,1.0,1.0,1.0,-2.0,-0.5,-0.3] # Control sequence
Δt = 1.0 # Time step

# First step
x = Vector{eltype(x_0)}(undef,3*(length(u)+1))
x[1:3] = x_0
for i ∈ 1:length(u)
    x[3*i + 1] = x[3*(i-1)+1] + V*cos(x[3*(i-1)+3])*Δt
    x[3*i + 2] = x[3*(i-1)+2] + V*sin(x[3*(i-1)+3])*Δt
    x[3*i + 3] = x[3*(i-1)+3] + u[i]*Δt
end

z = [x;u]