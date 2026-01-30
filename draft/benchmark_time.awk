# Print the division
# Call using `ls -lU --time-style=+%s /work/datasets/aliby_output/*/jump_core_annotated/jpegxl_lossy_mq.zarr/profiles |awk -f benchmark_time.awk`
BEGIN {printf "*Model* *sites/min* *Hours(total)*\n"};
/^\// {
    group_starts=NR;
    if (NR>1) names[s[5]]=max-min;
    split($0, s, "/");
    nfiles = 0;
} 
    {if (NR == group_starts+2) min = max = $6}
    /parquet/ {
	nfiles++;
	if ($6 < min) min = $6;
	if ($6 > max) max = $6;
    }
    END {
	names[s[5]]=max-min;
	for (key in names) printf("%s %s %s\n", key, nfiles/names[key]*60, names[key]/3600)}
