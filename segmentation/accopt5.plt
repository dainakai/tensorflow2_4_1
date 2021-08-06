set terminal pngcairo enhanced font 'Times New Roman,15'
set key right top box maxrows 5
set size ratio 0.6
set output "accopt5.png"
set xlabel 'Number of droplets {/Times-New-Roman:Italic=15 m} of tested dataset'
set ylabel 'Inference accuracy'
set ytics 0.5,0.1,1.0
set yrange[0.45:1.05]
set xrange[1.5:15.5]

plot "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num7.txt" using 1:2 title "n = 7" with linespoints lc "purple", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num8.txt" using 1:2 title "n = 8" with linespoints lc "orange", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num9.txt" using 1:2 title "n = 9" with linespoints lc "blue", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num10.txt" using 1:2 title "n = 10" with linespoints lc "red", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num11.txt" using 1:2 title "n = 11" with linespoints lc "dark-green"