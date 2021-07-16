set terminal pngcairo enhanced font 'Times New Roman,10'
unset key
set output "./processed/pro_close.png"
set multiplot
set size ratio 1

set origin 0,0
set size 0.5,1
set size ratio 1 
set xrange[0:1024]
set yrange[0:1024]
set xtics 200
set ytics 200
set tics out
set xlabel '{/Times-New-Roman:Italic=10 x} [pixel]'
set ylabel '{/Times-New-Roman:Italic=10 y} [pixel]'
plot "./raw/close6_cont.png" binary filetype=png origin=(0.5,0.5) with rgbimage, "prorectan.txt" using 1:2 with lines lc "black"

set nokey

set origin 0.5,0.001
set size 0.4855,1
set size ratio 1
set xrange[640:820]
set yrange[220:400]
set xlabel '{/Times-New-Roman:Italic=10 x} [pixel]'
set ylabel '{/Times-New-Roman:Italic=10 y} [pixel]'
set xtics 50
set ytics 50
set tics out
replot

unset multiplot