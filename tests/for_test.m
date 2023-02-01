s = 10;
H = zeros(s);

for c = 1:s
    for r = 1:s
        H(r,c) = 1/(r+c-1);
    end
end

disp(H)

for i =  [1,2,3;4,5,6;7,8,9]
    disp(i)
end