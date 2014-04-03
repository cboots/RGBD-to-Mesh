close all
clear all

y = 0;
x = -1:0.01:1;
z = -sqrt(1-y*y-x.*x);


figure
plot(x,z);

azimuth = atan2(-z,-x);
elv = acos(-y);
figure
plot(x, azimuth);

x2 = cos(azimuth)*sin(elv);
z2 = sin(azimuth)*sin(elv);
y2 = cos(elv);
figure
plot(azimuth,x2);
hold all
plot(azimuth,z2);