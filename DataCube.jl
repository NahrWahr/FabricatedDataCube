begin
    using Plots
    plotly()
    using LinearAlgebra
    using ForwardDiff
    using JLD2
end

begin
    const permutation = UInt8[
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233,
        7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
        190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
        203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174,
        20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27,
        166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
        220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
        63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
        200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
        186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
        126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182,
        189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
        70, 221, 153, 101, 155, 167,  43, 172, 9, 129, 22, 39, 253, 19,
        98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246,
        97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
        241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181,
        199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150,
        254, 138, 236, 205, 93, 222,    114, 67, 29, 24, 72, 243, 141,
        128, 195, 78, 66, 215, 61, 156, 180]

    function grad(h::Integer, x, y, z)
        h &= 15                                                 # CONVERT LO 4 BITS OF HASH CODE
        u = h < 8 ? x : y                                       # INTO 12 GRADIENT DIRECTIONS.
        v = h < 4 ? y : h == 12 || h == 14 ? x : z
        (h & 1 == 0 ? u : -u) + (h & 2 == 0 ? v : -v)
    end

    function perlinsnoise(x, y, z)
        p = vcat(permutation,  permutation)
        fade(t) = t * t * t * (t * (t * 6 - 15) + 10)
        lerp(t, a, b) = a + t * (b - a)
        floorb(x) = Int(floor(x)) & 0xff
        X, Y, Z = floorb(x), floorb(y), floorb(z)               # FIND UNIT CUBE THAT CONTAINS POINT.
        x, y, z = x - floor(x), y - floor(y), z - floor(z)      # FIND RELATIVE X,Y,Z OF POINT IN CUBE.
        u, v, w = fade(x), fade(y), fade(z)                     # COMPUTE FADE CURVES FOR EACH OF X,Y,Z.
        A = p[X + 1] + Y; AA = p[A + 1] + Z; AB = p[A + 2] + Z
        B = p[X + 2] + Y; BA = p[B + 1] + Z; BB = p[B + 2] + Z  # HASH COORDINATES OF THE 8 CUBE CORNERS

        return lerp(w, lerp(v, lerp(u,   grad(p[AA + 1], x  , y  , z       ),  # AND ADD
                                    grad(p[BA + 1], x - 1, y  , z    )),  # BLENDED
                            lerp(u,   grad(p[AB + 1], x  , y - 1, z     ),  # RESULTS
                                 grad(p[BB + 1], x - 1, y - 1, z ))),  # FROM  8
                    lerp(v, lerp(u, grad(p[AA + 2], x  , y  , z - 1   ),  # CORNERS
                                 grad(p[BA + 2], x - 1, y  , z - 1)),  # OF CUBE.
                         lerp(u, grad(p[AB + 2], x  , y - 1, z - 1 ),
                              grad(p[BB + 2], x - 1, y - 1, z - 1))))
    end
end

function CrTop(n,m,p=50,q=50,seed=rand(1:10))
    pnt = [[perlinsnoise(x/p,y/q,seed) for x=-n:n] for y=-m:m]
    #pnt = [[cos((x^2+y^2)/3000) for x=-n:n] for y=-m:m]
    top = hcat(pnt...)
    return top
end

top = CrTop(100,100)*10; surface(top, zlim=(-100,100))

gradx,grady = [diff(top, dims=1)[:,begin:end-1], diff(top, dims=2)[begin:end-1,:]]

begin
    a = []
    for i in eachindex(gradx)
	push!(a, [gradx[i], grady[i], 1])
    end
    a = reshape(a, (200,200))
end

begin
    sun = [0,0.5,0.3]
    inten = []
    for i in eachindex(a)
	push!(inten, a[i]'*sun)
    end
    outiten=reshape(inten, (200,200))
end

surface(outiten, zlim=(-10,10))

function CrEM(bands, seed = 10)
    coefs = randn(seed)
    freq = randn(seed)
    ar = [(x -> ((map(sin, freq*x))+map(cos, freq*x)))(i/10)'coefs for i=-bands:bands-1]
    return ar.^2/100
end

begin
    EMs = []
    for i=1:10
	push!(EMs, CrEM(100,300))
    end
end

plot(EMs; layout=(5,1), legend=false, colorbar=false, showaxis=false, ticks=false, )

function CrMix(n,m,EMs)
    Ar = []
    for i=1:n*m
	append!(Ar, sum(rand(length(EMs)) .* EMs))
    end
    Ar = permutedims(reshape(Ar,length(EMs[1]), n, m), (2,3,1))
    return Ar
end

result = CrMix(200,200,EMs) .* outiten

surface(result[:,:,10], zlim=(-100,100), )
