function[beta,  xtrain, ytrain, samples, features] = train(name)
    data = load(name);
    xtrain = data.Xtrain;
    ytrain = data.ytrain;
    x_norm = normalize(xtrain);
    [samples, features] = size(x_norm);
    beta = zeros(features,1);
    beta = grad_descent(beta, 1500, x_norm, ytrain)
endfunction

function[n] = normalize(mat) 
    n = z_norm(mat);
endfunction

function[n] = z_norm(mat)
    n = zscore(mat);
endfunction

function[n] = log_norm(mat)
    [x,y] = size(mat);
    n = zeros(x,y);
    for i = 1:x
        for j = 1:y
            n(i,j) = log(mat(i,j)+0.1);
        endfor
    endfor
endfunction

function[n] = binarize_norm(mat)
    [x,y] = size(mat);
    n = zeros(x,y);
    for i = 1:x
        for j = 1:y
            if mat(i,j) > 0
                n(i,j) = 1;
            else
                n(i,j) = 0;
            endif
        endfor
    endfor
endfunction

function[beta] = grad_descent(b, iter, x, y, rho=0.5,lambda=0.0)
    beta = b;
    for i = 1:iter
        beta = beta + rho*(x'*(y - arrayfun(@(x) 1/x,(1+exp(-x*b)))) + lambda*beta);
    endfor
endfunction
