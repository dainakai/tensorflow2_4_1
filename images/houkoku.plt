set terminal pngcairo enhanced font 'Times New Roman,15'
unset key
set size ratio 1
set xlabel '{/Times-New-Roman:Italic=20 x} [pixel]'
set ylabel '{/Times-New-Roman:Italic=20 y} [pixel]'
set xrange[0:5]
set yrange[0:5]
set tics out
set output "currentcnn_houkoku/conv0.png"
plot "currentcnn/conv0.png" binary filetype=png origin=(0.5,0.5) with rgbimage