%%testCalcNormal
% clear all
clc

A = [[1 2 1]; [2 3 3]; [ 1 3 4]]

[N c] = calcNormal([], A)


rdist = rand(100,3);
points = bsxfun(@times, rdist, [10 10 0.1]);
A = points'*points


[N c] = calcNormal([], A)
