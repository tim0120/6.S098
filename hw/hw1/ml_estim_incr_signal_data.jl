module MLEstim

    export h
    export _conv

    # create problem data
    using Random
    Random.seed!(0);
    N = 100;
    # create an increasing input signal
    xtrue = zeros(N);
    xtrue[1:40] .= 0.1;
    xtrue[50] = 2;
    xtrue[70:80] .= 0.15;
    xtrue[80] = 1;
    xtrue = cumsum(xtrue);

    # conv brought to you by github copilot
    function _conv(x, y)
        N = length(x)
        M = length(y)
        z = zeros(N+M-1)
        for n in 1:N
            for m in 1:M
                z[n+m-1] += x[n]*y[m]
            end
        end
        return z
    end

    # pass the increasing input through a moving-average filter
    # and add Gaussian noise
    h = [1, -0.85, 0.7, -0.3];
    k = length(h);
    yhat = _conv(h, xtrue);
    y = yhat[1:end-3] + randn(N);

    using Plots
    length(xtrue)
    length(yhat)
    plot(1:N, xtrue)
    plot!(1:N, y)

    using DataFrames
    using CSV
    y = vec(y)
    df = DataFrame(y=y, xtrue=xtrue)
    CSV.write("l_estim.txt", df)
end