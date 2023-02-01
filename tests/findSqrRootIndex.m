function idx = findSqrRootIndex(target,arrayToSearch)

idx = NaN;
if target < 0
   return
end

for idx = 1:length(arrayToSearch)
    if arrayToSearch(idx) == sqrt(target)
        return
    end
end