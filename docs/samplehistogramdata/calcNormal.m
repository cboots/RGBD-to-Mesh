function [v curve eigs] = calcNormal(points ,A)

if(nargin < 2)
    A = points'*points;
end

p1 = A(1,2)^2 + A(1,3)^2 + A(2,3)^2;
if (p1 == 0) 
   % A is diagonal.
   eig1 = A(1,1);
   eig2 = A(2,2);
   eig3 = A(3,3);
else
   q = trace(A)/3;
   p2 = (A(1,1) - q)^2 + (A(2,2) - q)^2 + (A(3,3) - q)^2 + 2 * p1;
   p = sqrt(p2 / 6);
   B = (1 / p) * (A - q * eye(3));       % I is the identity matrix
   r = det(B) / 2;
 
   % In exact arithmetic for a symmetric matrix  -1 <= r <= 1
   % but computation error can leave it slightly outside this range.
   if (r <= -1) 
      phi = pi / 3;
   elseif (r >= 1)
      phi = 0;
   else
      phi = acos(r) / 3;
   end
 
   % the eigenvalues satisfy eig3 <= eig2 <= eig1
   eig1 = q + 2 * p * cos(phi);
   eig3 = q + 2 * p * cos(phi + (2*pi/3));
   eig2 = 3 * q - eig1 - eig3;     % since trace(A) = eig1 + eig2 + eig3
end

mineig = min([eig1 eig2 eig3]);


[V D] = eig(A);

N = (A-eye(3)*eig1)*(A(:,1)-[1;0;0]*eig2);



norms = sqrt(sum(N.^2,1));
[~,i] = max(norms);
if(min(norms) == 0)
    stop
end
v = N(:,i)./norms(i);
curve = eig3/(eig1+eig2+eig3);
eigs = [eig1 eig2 eig3];