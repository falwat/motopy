limit = 0.75;
A = rand(10,1)

if any(A > limit)
    disp('There is at least one value above the limit.')
else
    disp('All values are below the limit.')
end

%%
x = 10;
if x ~= 0
    disp('Nonzero value')
end

%% 
x = 10;
minVal = 2;
maxVal = 6;

if (x >= minVal) && (x <= maxVal)
    disp('Value within specified range.')
elseif (x > maxVal)
    disp('Value exceeds maximum value.')
else
    disp('Value is below minimum value.')
end

%%
nrows = 4;
ncols = 6;
A = ones(nrows,ncols);

for c = 1:ncols
    for r = 1:nrows
        
        if r == c
            A(r,c) = 2;
        elseif abs(r-c) == 1
            A(r,c) = -1;
        else
            A(r,c) = 0;
        end
        
    end
end