close all
clear all

%% Import the data
%data = xlsread('decoupledhistogrammixedscene.csv');
%data = xlsread('decoupledhistogramflatwallheadon.csv');
data = xlsread('decoupledhistogramuppercorner.csv');
%data = xlsread('decoupledhistogramceiling.csv');
%data = xlsread('decoupledhistogramnoplanes.csv');

%% Allocate imported array to column variable names
normXHist = data(:,1);
normYHist = data(:,2);
normZHist = data(:,3);

%% Clear temporary variables
clearvars data raw columnIndices;

bins = [1:512]';

figure
plot(bins, normXHist, bins, normYHist, bins, normZHist);
title('histogram');

testXHist = normXHist;
testYHist = normYHist;
testZHist = normZHist;

range = 64;
%%X histogram
[peakV peakI] = max(testXHist);
sigma = 20;
peaksX = [];

figure
plot(bins,testXHist);
hold all
while peakV > 500
    peaksX = [peaksX peakI];
    gauss = peakV * exp( - (bins-peakI).*(bins-peakI) / (2*sigma*sigma));
    updaterange = unique(min(max(peakI-range:peakI+range,1),512));
    testXHist(updaterange) = testXHist(updaterange) - gauss(updaterange);
    
    plot(bins,gauss);
    plot(bins,testXHist);
    
    [peakV peakI] = max(testXHist);
end

length(peaksX)



%%Y histogram
[peakV peakI] = max(testYHist);
sigma = 20;
peaksY = [];

figure
plot(bins,testYHist);
hold all
while peakV > 500
    peaksY = [peaksY peakI];
    gauss = peakV * exp( - (bins-peakI).*(bins-peakI) / (2*sigma*sigma));
    testYHist = testYHist - gauss;
    
    plot(bins,gauss);
    plot(bins,testYHist);
    
    [peakV peakI] = max(testYHist);
end

length(peaksY)



%%Z histogram
[peakV peakI] = max(testZHist);
sigma = 25;
peaksZ = [];

figure
plot(bins,testZHist);
hold all
while peakV > 500
    peaksZ = [peaksZ peakI];
    gauss = peakV * exp( - (bins-peakI).*(bins-peakI) / (2*sigma*sigma));
    testZHist = testZHist - gauss;
    
    plot(bins,gauss);
    plot(bins,testZHist);
    
    [peakV peakI] = max(testZHist);
end

length(peaksZ)
