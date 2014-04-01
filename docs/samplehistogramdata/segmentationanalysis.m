close all

if (~exist('segmentDistances'))
   
%% Import the data
% data = xlsread('segmentationSampleCorner2.csv');
% data = xlsread('segmentationSampleCabinet.csv');
% data = xlsread('segmentationSampleBackwall.csv');
data = xlsread('segmentationSampleTiltedWhiteboard.csv');

%% Allocate imported array to column variable names
posX = reshape(data(:,1),[640 480])';
posY = reshape(data(:,2),[640 480])';
posZ = reshape(data(:,3),[640 480])';
normX= reshape(data(:,4),[640 480])';
normY= reshape(data(:,5),[640 480])';
normZ= reshape(data(:,6),[640 480])';
segmentIndexes = reshape(data(:,7),[640 480])';
segmentDistances = reshape(data(:,8),[640 480])';
red = zeros(480,640);
grn = zeros(480,640);
blu = zeros(480,640);
if(size(data,2) > 8)
    red = reshape(data(:,9),[640 480])';
    grn = reshape(data(:,10),[640 480])';
    blu = reshape(data(:,11),[640 480])';
end

rgbIm = cat(3, red,grn,blu);

%% Clear temporary variables
clearvars data raw columnIndices red grn blu;
 
end

figure
image(rgbIm);

figure
imagesc(segmentIndexes);

figure
imagesc(segmentDistances);

allSegments = zeros(480,640,4);
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
       segments(mask & (abs(pkcenter - segmentDistances) < 0.01)) = j;
       
    end

    subplot(2,2,4)
    imagesc(segments);
    
    distances = zeros(length(locs),1);
    normals = zeros(length(locs),3);
    counts = zeros(length(locs),1);
    centroids = zeros(length(locs),3);
    for j=1:length(locs)
        points = [posX(segments == j) posY(segments == j) posZ(segments == j)];
        if(size(points,1) > 300)
            
        cm = mean(points,1);
        points0 = bsxfun(@minus, points, cm);
        
        [U,S,V] = svd(points0,0);
        
        disp('Normals:')
        normal = V(:,3);
        [normalOther curvature] = calcNormal(points0);
        normal'
        normalOther'
        curvature
        
        
        if(normal(3) < 0.0)
            normal = -normal;
        end
        normals(j,:) = normal;
        counts(j) = size(points,1);
        distances(j) = dot(normal,cm);
        centroids(j,:) = cm;
        
        end
    end
    
    
    
    avgnorm = sum(bsxfun(@times, counts, normals),1)/sum(counts);
    avgnorm = avgnorm./sqrt(sum(avgnorm.^2))
    
    acosd(normals(1,:)*avgnorm')
    avgnorm = normals(1,:)
    newprojection = mask.*(posX*avgnorm(1) + posY*avgnorm(2)+ posZ*avgnorm(3));
    newprojection(newprojection == 0) = nan;
    
    figure
    subplot(2,1,1)
    surf(newprojection);
    view([0 0])
    
    counts = hist(newprojection(:), samples);
    subplot(2,1,2)
    plot(samples, counts);
    
    
    %%
    
    
    %%
    
    anglethresh =  cos(2.5*pi/180.0);
    distthresh = 0.02;
    
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
    merge = (delts < distthresh) & (dotproducts > anglethresh);
    
    finalsegments = zeros(480,640);
    for x=1:size(merge,1)
        matches = find(merge(x,:));
        for y=matches
            finalsegments(segments==y) = x;
        end
    end
    
    figure
    imagesc(finalsegments);
    
    
    
    %% Compute final segmentation
    segmentids = unique(finalsegments);
    segmentids(segmentids==0) = [];
    
    anglethresh = cosd(25.0);
    distthresh = 0.025;
    fullImageSegmentation = zeros(480,640);
    
    points = cat(3,posX, posY, posZ);
    norms = cat(3,normX, normY, normZ);
    
    for seg=1:length(segmentids)
        j = segmentids(seg);
        norm = normals(j,:);
        d = distances(j);
        
        normdotprod = abs(bsxfun(@times, norms(:,:,1), norm(1)) ...
                + bsxfun(@times, norms(:,:,2), norm(2)) ...
                + bsxfun(@times, norms(:,:,3), norm(3)));
            
        dist = bsxfun(@times, points(:,:,1), norm(1)) ...
                + bsxfun(@times, points(:,:,2), norm(2)) ...
                + bsxfun(@times, points(:,:,3), norm(3));
        
            
        anglemask = normdotprod > anglethresh;
        distmask = abs(dist - d) < distthresh; 
        
        
%         figure
%         subplot(2,2,1)
%         imagesc(normdotprod);
%         
%         subplot(2,2,2)
%         imagesc(abs(dist-d));
%         
%         subplot(2,2,3);
%         imagesc(anglemask);
%         
%         subplot(2,2,4);
%         imagesc(distmask);
            
        %%Don't overwrite previous segments
        fullImageSegmentation(anglemask & distmask & (fullImageSegmentation == 0)) = j;
    end
    
    figure
    imagesc(fullImageSegmentation);
    
    allSegments(:,:,i+1) = fullImageSegmentation;
end

end


%% Assemble final 
segments1 = unique(allSegments(:,:,1));
segments1 = segments1(segments1 > 0);
segments2 = unique(allSegments(:,:,2));
segments2 = segments2(segments2 > 0);
segments3 = unique(allSegments(:,:,3));
segments3 = segments3(segments3 > 0);
segments4 = unique(allSegments(:,:,4));
segments4 = segments4(segments4 > 0);

start = 1;
segmentsmap1 = start:start+length(segments1);
start = max(segmentsmap1)+1;

segmentsmap2 = start:start+length(segments2);
start = max(segmentsmap2)+1;

segmentsmap3 = start:start+length(segments3);
start = max(segmentsmap3)+1;


segmentsmap4 = start:start+length(segments4);

for i = 1:length(segments1)
    oldId = segments1(i);
    segs = allSegments(:,:,1);
    segs(segs==oldId) = segmentsmap1(i);
    allSegments(:,:,1) = segs;
end

for i = 1:length(segments2)
    oldId = segments2(i);
    segs = allSegments(:,:,2);
    segs(segs==oldId) = segmentsmap2(i);
    allSegments(:,:,2) = segs;
end

for i = 1:length(segments3)
    oldId = segments3(i);
    segs = allSegments(:,:,3);
    segs(segs==oldId) = segmentsmap3(i);
    allSegments(:,:,3) = segs;
end

for i = 1:length(segments4)
    oldId = segments4(i);
    segs = allSegments(:,:,4);
    segs(segs==oldId) = segmentsmap4(i);
    allSegments(:,:,4) = segs;
end

mergedImage = allSegments(:,:,1) + allSegments(:,:,2) + ...
                allSegments(:,:,3) + allSegments(:,:,4);
            
         
figure
imagesc(mergedImage);
