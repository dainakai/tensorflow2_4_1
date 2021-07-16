set terminal pngcairo enhanced font 'Times New Roman,15'
set key right top box maxrows 5
set size ratio 0.6
set output "acc.png"
set xlabel 'Number of droplets {/Times-New-Roman:Italic=15 m} of tested dataset'
set ylabel 'Inference accuracy'
set ytics 0.5,0.1,1.0
set yrange[0.45:1.05]
set xrange[1.5:15.5]

plot "2.txt" using 2:3 title "n = 2" with linespoints, "3.txt" using 2:3 title "n = 3" with linespoints, "4.txt" using 2:3 title "n = 4" with linespoints, "5.txt" using 2:3 title "n = 5" with linespoints, "6.txt" using 2:3 title "n = 6" with linespoints, "7.txt" using 2:3 title "n = 7" with linespoints, "8.txt" using 2:3 title "n = 8" with linespoints, "9.txt" using 2:3 title "n = 9" with linespoints, "10.txt" using 2:3 title "n = 10" with linespoints