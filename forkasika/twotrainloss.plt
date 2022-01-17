set terminal pdfcairo enhanced font 'Times New Roman,15'
set key right top box
set key width -2
set size ratio 1
set output "loss.pdf"
set xrange[0.5:17.5]
set mxtics 2
set mytics 2
set xtics 1,2,17
set xlabel 'Epochs [-]'
set ylabel 'Loss value'

plot "trainloss.txt" using ($0+1):1 with linespoints title "train loss", "valloss.txt" using ($0+1):1 with linespoints title "validation loss"

unset output