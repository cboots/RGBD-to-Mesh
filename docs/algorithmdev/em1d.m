clear all
close all
 
x = -10:0.01:10;
sigma = 1;

gauss = exp(-x.*x/(2*sigma));

figure
plot(x,gauss);
hold all
for i=[1:10]
   plot(x,i*gauss); 
end
hold off

stop


bins = zeros(1, 512);

trueclustermeans =     [100 256 300 375 450];
trueclustersigmas=     [1000 2 1000 200 100];
trueclusterpeakcounts= [1000 200 900 600 500];
noise=0.1;

%Add in point clusters
ind = 1:512;
for i=1:length(trueclustermeans)
    diff = ind-trueclustermeans(i);
    expont = -diff.*diff/(2*trueclustersigmas(i));
    bins = bins + trueclusterpeakcounts(i)*exp(expont);
end

figure
plot(bins);
axis([1 512 0 5000])
title('Noiseless Histogram');

%Generate noise
bins = bins.*(1 + noise*rand(1, 512));


figure
plot(bins);
axis([1 512 0 5000])
title('Noisy histogram');

cdf = cumsum(bins./sum(bins));
figure
plot(cdf);
title('Cumulative Distribution Function')

%Map uniform distribution to cdf
px = [0.01:0.01:1]';
invmap = interp1q(cdf', ind', px);

px = [0; px; 1];
invmap = [1; invmap; 512];

figure
plot(px, invmap);
title('Inverse map');

n = 10000;
points = interp1q(px, invmap, rand(n, 1));

nelements = hist(points, 512);

%% Clustering Algorithm
k = 10;
idx = kmeans(points,k);

kmeansmin = ones(k, 1);

figure
for i=1:k
    kmeansmin(i) = min(points(idx == i));
    plot(points(idx == i),'.')
    hold all
end
hold off

kmeansmin = sort(kmeansmin);
ltgrid = bsxfun(@gt, kmeansmin, ind)';

[sel, c] = max( ltgrid ~=0, [], 2 );

figure
scatter(ind, nelements, 5.0, c);

options = statset('Display', 'final');
obj = gmdistribution.fit(points,k, 'Options', options);
x = 1:512;
x = x';
figure
plot(pdf(obj, x));
title('GMM Reconstruction');

%%
minsize = bins > 100;

diff = sign(bins(2:512)-bins(1:511));
graddir = [0 diff(1:510)+diff(2:511) 1];

