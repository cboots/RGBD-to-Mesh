close all
clear X map;
imglist = {'flujet', ... Fluid Jet
           'spine', ... Bone
           'gatlin', ... Gatlinburg
           'durer', ... Durer
           'detail', ... Durer Detail
           'cape', ... Cape Cod
           'clown', ... Clown
           'earth', ... Earth
           'mandrill', ... Mandrill
           'spiral'};

colorlabels = {'default', 'hsv','hot','pink',...
               'cool','bone','prism','flag',...
               'gray','rand'};

load(imglist{4},'X','map');
imagesc(X);
colormap(map);
%colormap(colorlabels{1});
axis off;


%sourcepoints = [200 150; 200 300;  400 300; 400 150;];
sourcepoints = [250 45; 305 22;  305 215; 250 190;];
hold on
plot(sourcepoints(:,1), sourcepoints(:,2));
hold off

destWidth = 50;
destHeight = 50;
destpoints = [1 1; destWidth 1; destWidth destHeight; 1 destHeight;];

a = [sourcepoints(1:3,1)'; sourcepoints(1:3,2)'; ones(1,3)]
b = [sourcepoints(4,:)'; 1]
x = linsolve(a, b);

A = bsxfun(@times, a, x');

a = [destpoints(1:3,1)'; destpoints(1:3,2)'; ones(1,3)];
b = [destpoints(4,:)'; 1];
x = linsolve(a, b);

B = bsxfun(@times, a, x');

C = A/B;

xd = repmat(1:destWidth, destHeight, 1);
yd = repmat(1:destHeight, destWidth, 1)';


homoSource = C*[xd(:) yd(:) ones(destHeight*destWidth,1)]';

sx = floor(homoSource(1,:)./homoSource(3,:));
sy = floor(homoSource(2,:)./homoSource(3,:));

linInd = sub2ind(size(X), sy, sx);
intensity = X(linInd);

I = reshape(intensity, [destHeight, destWidth]);

figure
imagesc(I);
colormap(map);
%colormap(colorlabels{1});
axis off;
