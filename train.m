function[beta,  xtrain, ytrain, samples, features] = train(data,normfunc)
    xtrain = data.Xtrain;
    ytrain = data.ytrain;
    printf("Normalizing data...\n");
    x_norm = nmz(xtrain, normfunc);
    [samples, features] = size(x_norm);
    beta = zeros(features,1);
    printf("Fitting parameters...\n");
    [beta, err, iters]= grad_descent(beta, 0.0001, x_norm, ytrain);
    printf("Fit paramaters in %d iterations\n", iters);
endfunction

function[n] = nmz(mat, normfunc)
    n = feval(normfunc, mat);
endfunction

function[n] = z_norm(mat)
    n = zscore(mat);
endfunction

function[n] = log_norm(mat)
    n = arrayfun(@(x) log(0.1+x), mat);
endfunction

function[n] = binarize_norm(mat)
    n = mat > 0;
endfunction

function[beta, err, i] = grad_descent(b, e, x, y, r=0.00001,l=0.0000001, fl=@(x,i) x)
    beta = b;
    i = 0;
    err = b;
    do
        temp = err;
        err = x'*(y - logistic(x, beta));
        change = norm(err) - norm(temp);
        gradient = (err + l*beta);
        beta = beta + r/(1)*gradient;
        i = i+1;
        printf('%f\n',norm(err));
    until (norm(err) < 70)
endfunction

function[p] = logistic(x,w)
    p = arrayfun(@(x) 1/x, (1 + exp(-x*w)));
endfunction
