set terminal pngcairo enhanced font 'Times New Roman,15'
unset key
set size ratio 0.6
set output "outout.png"

set xlabel 'Number of droplets [-]'
set ylabel 'Test data accuracy [-]'
# set xrange[0:12]
set yrange[0.5:1.05]
set ytics 0.5,0.1,1.0
set xtics 1,1,11
set style boxplot fraction 1
plot "test.txt" using (1.0):2:(0):1 with boxplot lc "dark-magenta", "avr.txt" using 1:2 with points lc "dark-magenta" pointtype 1

