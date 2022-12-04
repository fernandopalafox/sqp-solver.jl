# Generate Dubins car trajectory given a control sequence and initial state

x_0 = [0.0,0.0,0.0,0.0] # Initial state
T = 3 # Number of states after initial state
u_0 = rand(2) # Initial control sequence
u = rand(2*(T-1)) # Control sequence
u_0 = [0;1] # Initial control sequence
u   = [0;0;0;-1]
Δt = 1 # Time step

# First step
x = Vector{eltype(x_0)}(undef,4*(T))
x[1] = x_0[1] + x_0[4]*cos(x_0[3])*Δt
x[2] = x_0[2] + x_0[4]*sin(x_0[3])*Δt
x[3] = x_0[3] + u_0[1]*Δt
x[4] = x_0[4] + u_0[2]*Δt
for i ∈ 2:T
    x[4*(i-1) + 1] = x[4*(i-2)+1] + x[4*(i-2)+4]*cos(x[4*(i-2)+3])*Δt
    x[4*(i-1) + 2] = x[4*(i-2)+2] + x[4*(i-2)+4]*sin(x[4*(i-2)+3])*Δt
    x[4*(i-1) + 3] = x[4*(i-2)+3] + u[2*(i-2)+1]*Δt
    x[4*(i-1) + 4] = x[4*(i-2)+4] + u[2*(i-2)+2]*Δt
end

z = [x;u_0;u];