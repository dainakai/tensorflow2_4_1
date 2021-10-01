set terminal pdfcairo enhanced font 'Times New Roman,15' xffffff
set key right top box maxrows 5
set size ratio 0.6
set output "./pdf/2.pdf"
set xlabel 'Number of droplets {/Times-New-Roman:Italic=15 m} of tested dataset'
set ylabel 'Inference accuracy'
set ytics 0.5,0.1,1.0
set yrange[0.45:1.05]
set xrange[1.5:15.5]

plot "2.txt" using 2:3 title "{/Times-New-Roman:Italic=15 n} = 2" with linespoints




unset output