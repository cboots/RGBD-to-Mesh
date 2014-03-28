close all

if (~exist('segmentDistances'))
   
%% Import the data
data = xlsread('segmentationSampleCorner2.csv');

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

    
    mask = zeros(480,640);
    mask(segmentIndexes == i) = 1;
    
    
    %current segment
    figure
    subplot(2,2,1)
    imagesc(mask)

    subplot(2,2,2)
    surf(segmentDistances.*mask);
    view([0 0])

    samples = logspace(log10(0.1), log10(5.0), 512);
    KS = hist(distancesegment, samples);
    segments = zeros(480,640);

    pks = [];
    locs = [];
    windowr = 2;
    KSCopy = KS;
    while(max(KSCopy) > 350)
       [peak loc] = max(KSCopy);
       pks = [pks peak];
       locs = [locs loc];
       KSCopy(loc-windowr:loc+windowr) = 0;
    end
    
    
    subplot(2,2,3)
    plot(samples, KS);
    hold all
    plot(samples(locs), pks, '*');
    
    for j=1:length(locs)
       pkcenter = samples(locs(j));
       segments(mask & (abs(pkcenter - segmentDistances) < 0.015)) = j;
       
    end

    subplot(2,2,4)
    imagesc(segments);
    
    distances = zeros(length(locs),1);
    normals = zeros(length(locs),3);
    counts = zeros(length(locs),1);
    centroids = zeros(length(locs),3);
    for j=1:length(locs)
        points = [posX(segments == j) posY(segments == j) posZ(segments == j)];
        cm = mean(points,1);
        points0 = bsxfun(@minus, points, cm);
        [U,S,V] = svd(points0,0);
        normal = V(:,3);
        if(normal(3) < 0.0)
            normal = -normal;
        end
        normals(j,:) = normal;
        counts(j) = size(points,1);
        distances(j) = dot(normal,cm);
        centroids(j,:) = cm;
    end
    
    normals;
    counts;
    distances;
    centroids;
    
    anglethresh =  cos(2.5*pi/180.0);
    distthresh = 0.015;
    
    dotproducts = (normals*normals');
    acosd(dotproducts);
    delts = zeros(size(dotproducts));
    for x = 1:size(dotproducts,1)
       for y = 1:size(dotproducts,1)
           d1 = abs(dot(centroids(x,:)-centroids(y,:), normals(x,:)));
           d2 = abs(dot(centroids(y,:)-centroids(x,:), normals(y,:)));
           delts(x,y) = min(d1,d2);
       end
    end
    delts;
    merge = (delts < distthresh) & (dotproducts > anglethresh)
    
    finalsegments = zeros(480,640);
    for x=1:size(merge,1)
        matches = find(merge(x,:));
        for y=matches
            finalsegments(segments==y) = x;
        end
    end
    
    figure
    imagesc(finalsegments);
    
    newprojection = mask.*(posX*normals(1,1) + posY*normals(1,2)+ posZ*normals(1,3));
    newprojection(newprojection == 0) = nan;
    
    figure
    subplot(2,1,1)
    surf(newprojection);
    view([0 0])
    
    counts = hist(newprojection(:), samples);
    subplot(2,1,2)
    plot(samples, counts);
    
    
    %% Compute final segmentation
    segmentids = unique(finalsegments);
    segmentids(segmentids==0) = [];
    
    anglethresh = cosd(30.0);
    distthresh = 0.025;
    fullImageSegmentation = zeros(480,640);
    
    points = cat(3,posX, posY, posZ);
    norms = cat(3,normX, normY, normZ);
    
    for seg=1:length(segmentids)
        j = segmentids(seg);
        norm = normals(j,:);
        d = distances(j);
        
        figure
        normdotprod = bsxfun(@times, norms(:,:,1), norm(1)) ...
                + bsxfun(@times, norms(:,:,2), norm(2)) ...
                + bsxfun(@times, norms(:,:,3), norm(3));
            
        dist = bsxfun(@times, points(:,:,1), norm(1)) ...
                + bsxfun(@times, points(:,:,2), norm(2)) ...
                + bsxfun(@times, points(:,:,3), norm(3));
        
            
        anglemask = normdotprod > anglethresh;
        distmask = abs(dist - d) < distthresh; 
        
        subplot(2,2,1)
        imagesc(normdotprod);
        
        subplot(2,2,2)
        imagesc(dist);
        
        subplot(2,2,3);
        imagesc(anglemask);
        
        subplot(2,2,4);
        imagesc(distmask);
            
        fullImageSegmentation(anglemask & distmask) = j;
    end
    
    figure
    imagesc(fullImageSegmentation);
    
    
end

end