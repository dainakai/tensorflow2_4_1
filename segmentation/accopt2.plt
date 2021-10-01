set terminal pdfcairo enhanced font 'Times New Roman,15'
set key left bottom box maxrows 5
set size ratio 0.6
set output "accopt2.pdf"
set xlabel 'Number of droplets {/Times-New-Roman:Italic=15 m} of tested dataset'
set ylabel 'Inference accuracy'
set ytics 0.5,0.1,1.0
set yrange[0.45:1.05]
set xrange[1.5:14.5]

plot "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num2.txt" using 1:2 title "n = 2" with linespoints lc "purple", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num3.txt" using 1:2 title "n = 3" with linespoints lc "orange", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num4.txt" using 1:2 title "n = 4" with linespoints lc "blue", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num5.txt" using 1:2 title "n = 5" with linespoints lc "red", "/home/dai/Documents/tensorflow2_4_1/segmentation/accdata/acc_opt2_seg2_num6.txt" using 1:2 title "n = 6" with linespoints lc "dark-green"

unset output