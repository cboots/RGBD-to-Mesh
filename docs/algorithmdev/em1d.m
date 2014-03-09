 clear all
 close all
 
 
 bins = zeros(1, 512);
 
 trueclustermeans =     [100 256 300 375 450];
 trueclustersigmas=     [1000 2 1000 200 100];
 trueclusterpeakcounts= [1000 200 900 600 500];
 noise=0.1;
 
 
 ind = 1:512;
 for i=1:length(trueclustermeans)
    diff = ind-trueclustermeans(i);
    expont = -diff.*diff/(2*trueclustersigmas(i));
    bins = bins + trueclusterpeakcounts(i)*exp(expont);
 end
 
 figure
 plot(bins);
 axis([1 512 0 5000])
 
 bins = bins.*(1 + noise*rand(1, 512));
 
 figure
 plot(bins);
 axis([1 512 0 5000])