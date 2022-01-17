set terminal pdfcairo enhanced font 'Times New Roman,15'
set key right bottom box
set key width -2
set size ratio 1
set output "acc.pdf"
set xrange[0.5:17.5]
set yrange[0.5:1.05]
set mxtics 2
set mytics 2
set xtics 1,2,17
set ytics 0.5,0.1,1.0
set xlabel 'Epochs [-]'
set ylabel 'Accuracy value'

plot "trainacc.txt" using ($0+1):1 with linespoints title "train accuracy", "valacc.txt" using ($0+1):1 with linespoints title "validation accuracy"
unset output