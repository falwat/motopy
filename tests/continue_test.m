for n = 1:50
    if mod(n,7) ~= 0
        continue
    end
    disp(['Divisible by 7: ', num2str(n)])
end