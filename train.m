function[data, xtrain, ytrain] = train(name)
    data = load(name);
    xtrain = data.Xtrain;
    ytrain = data.ytrain;
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
