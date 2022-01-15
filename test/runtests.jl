using titanic
using Test

@testset "titanic.jl" begin

    @testset "logistic regression" begin
        X = [1.0 1.0 1.0; 1.0 2.0 3.0]
        k = [1, -1, -1]
        w = [1.5, -0.5]

        @test isapprox(titanic.computeE(X, k, w), 0.66016; atol=1e-5)

        @test all(isapprox.(titanic.gradE(X, k, w), [0.28450, 0.82532]; atol=1e-5))

        @testset "||$(x)|| = √2" for x in [[1, 1], [-1, -1], [-1, 1], [1, -1]]
            @test isapprox(titanic.l2norm(x), sqrt(2); atol=1e-5) 
        end

        @test all(titanic.classify(X, [2.0, -1.0]) .== [1, 1, -1])

    end

end
