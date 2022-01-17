set terminal pdfcairo enhanced font 'Times New Roman,15'
unset key
set size ratio 1
set output "t0.pdf"
set xrange[0:61]
set yrange[0:61]
set xtics 0,10,61
set ytics 0,10,61
set xlabel '{/Times-New-Roman:Italic=20 x} [pixel]'
set ylabel '{/Times-New-Roman:Italic=20 y} [pixel]'

plot "t0weights.txt" matrix using 1:2:3 with image
unset output