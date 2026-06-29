using LinearAlgebra

J = randn(3, 9)
f = randn(3)
F = qr(J', ColumnNorm())

P = F.P
z = zeros(3)
δgn = zeros(9)
ldiv!(z, LowerTriangular(F.R'), P'*(-f))
mul!(δgn, Matrix(F.Q), z)
P = F.P
P'P

B = A*A'

F = eigen(Symmetric(B))
Q = F.vectors
Λ = F.values
Q*diagm(Λ)*Q'


C = copy(B)
C[1, 1] = 0.0

FC = eigen(Symmetric(C))
Q = FC.vectors
Λ = FC.values
Q*diagm(Λ)*Q'

f = rand(3)

z = zeros(3)

ldiv!(z, F, -f)
