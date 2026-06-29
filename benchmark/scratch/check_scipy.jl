using Pkg
Pkg.activate(@__DIR__)
using PythonCall
try
    scipy = pyimport("scipy")
    println("Success: scipy imported.")
catch e
    println("Error: $e")
    exit(1)
end
