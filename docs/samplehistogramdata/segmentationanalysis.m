close all

if (~exist('segmentDistances'))
   
%% Import the data
data = xlsread('segmentationSampleShelves.csv');

%% Allocate imported array to column variable names
posX = reshape(data(:,1),[640 480])';
posY = reshape(data(:,2),[640 480])';
posZ = reshape(data(:,3),[640 480])';
normX= reshape(data(:,4),[640 480])';
normY= reshape(data(:,5),[640 480])';
normZ= reshape(data(:,6),[640 480])';
segmentIndexes = reshape(data(:,7),[640 480])';
segmentDistances = reshape(data(:,8),[640 480])';

%% Clear temporary variables
clearvars data raw columnIndices;
 
end


figure
imagesc(segmentIndexes);

figure
imagesc(segmentDistances);

for i=0:3
distancesegment = segmentDistances(segmentIndexes == i);
distancesegment(distancesegment < 0) = [];

count = length(distancesegment);
if(count > 500)

    %current segment
    figure
    mask = zeros(480,640);
    mask(segmentIndexes == i) = 1;
    subplot(2,2,1)
    imagesc(mask)

    subplot(2,2,2)
    surf(segmentDistances.*mask);
    view([0 0])

    samples = logspace(log10(0.1), log10(5.0), 512);
    KS = hist(distancesegment, samples);
    [pks locs] = findpeaks(KS);


    subplot(2,2,3)
    plot(samples, KS);
    hold all
    plot(samples(locs), pks, '*');

    segments = zeros(480,640);

    [pks I] = sort(pks, 'ascend');
    locs = locs(I);

    windowr = 2;
    for j=1:length(locs)
        pkcenter = samples(locs(j));
       
            segments(mask & (abs(pkcenter - segmentDistances) < 0.0075)) = j;
       
    end

    subplot(2,2,4)
    imagesc(segments);
end

end