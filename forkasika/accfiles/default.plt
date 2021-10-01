set terminal pdfcairo enhanced font 'Times New Roman,15'
set key right top box maxrows 5
set size ratio 0.6
set output "./pdf/6.pdf"
set xlabel 'Number of droplets {/Times-New-Roman:Italic=15 m} of tested dataset'
set ylabel 'Inference accuracy'
set ytics 0.5,0.1,1.0
set yrange[0.45:1.05]
set xrange[1.5:15.5]

plot "2.txt" using 2:3 title "{/Times-New-Roman:Italic=15 n} = 2" with linespoints, "3.txt" using 2:3 title "{/Times-New-Roman:Italic=15 n} = 3" with linespoints lc 'orange',"4.txt" using 2:3 title "{/Times-New-Roman:Italic=15 n} = 4" with linespoints lc 'royalblue',"5.txt" using 2:3 title "{/Times-New-Roman:Italic=15 n} = 5" with linespoints lc 'red',"6.txt" using 2:3 title "{/Times-New-Roman:Italic=15 n} = 6" with linespoints lc 'forest-green',




unset output