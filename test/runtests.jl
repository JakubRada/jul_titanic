using titanic
using Test
using titanic: WeakClassifier, Labels, Dataset, classificationerror, out, computeE, gradE, l2norm, linearkernel, polynomialkernel, gaussiankernel

@testset "titanic.jl" begin

    @testset "adaboost" begin
        x1 = [1, 2, 3, 4]
        x2 = [5, 6, 7, 8]
        h1 = WeakClassifier(1, 3, true)
        h2 = WeakClassifier(2, 2, true)
        h3 = WeakClassifier(3, 4, false)
        h4 = WeakClassifier(4, 8, false)

        @test h1(x1) == -1
        @test h1(x2) == 1

        @test h2(x1) == 1
        @test h2(x2) == 1

        @test h3(x1) == 1
        @test h3(x2) == -1

        @test h4(x1) == 1
        @test h4(x2) == -1
    end

    @testset "dataset" begin
        prediction = Labels([1, -1, 1, 1, -1, 1, 1, 1, -1, -1])
        truth = Labels([-1, -1, -1, 1, 1, 1, -1, 1, 1, -1])

        @test classificationerror(truth, prediction) == 0.5

        @test all(out(truth)[2] .== [0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
    end

    @testset "logistic regression" begin
        X = [1.0 1.0 1.0; 1.0 2.0 3.0]
        k = [1, -1, -1]
        w = [1.5, -0.5]

        @test isapprox(computeE(X, k, w), 0.66016; atol=1e-5)

        @test all(isapprox.(gradE(X, k, w), [0.28450, 0.82532]; atol=1e-5))

        @testset "||$(x)|| = √2" for x in [[1, 1], [-1, -1], [-1, 1], [1, -1]]
            @test isapprox(l2norm(x), sqrt(2); atol=1e-5) 
        end

        @test all(classify(Dataset(X), [2.0, -1.0])() .== [1, 1, -1])

    end

    @testset "svm" begin
        x1 = [1, 2, 3, 4, 5]
        x2 = [1, 1, 1, 1, 1]

        @test linearkernel(x1, x2) == 15
        @test polynomialkernel(x1, x2; degree=1) == 16
        @test polynomialkernel(x1, x2; degree=2) == 256
        @test isapprox(gaussiankernel(x1, x2; variance=1.0), 3.1e-7; atol=1e-8)
    end

end
