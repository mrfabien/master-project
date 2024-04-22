storm = readtable('/Users/fabienaugsburger/Documents/GitHub/master-project/tracks_square_storm/storm_data.csv')

lon_east = storm(:,1);
diff_lon_east = abs(diff(lon_east));

min_lon_diff = min(diff_lon_east)
max_lo_diff = max(diff_lon_east)

for i=1:height(diff_lon_east(:,1))
    if diff_lon_east(i,1)<10
        diff_modified = diff_lon_east(i)
    end
end

