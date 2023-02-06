dif = subtract(2, 1)

absdif = abs(subtract(2, 1))

dif, absdif = subtract(2, 1)

function [dif,absdif] = subtract(y,x)
    dif = y-x;
    if nargout > 1
        disp('Calculating absolute value')
        absdif = abs(dif);
    end
end