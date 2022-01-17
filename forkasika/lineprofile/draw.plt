# set terminal pngcairo enhanced font 'Times New Roman,15'
# # set size ratio 0.3
# set output "lineprofile.png"
# set key right top box
# set lmargin 10
# set offsets at screen 0,0,0,2
# # set bmargin 4
# # set tmargin 2
# set yrange[0.94:1.06]
# set multiplot layout 2,1
# set label 1 "Distance from the center of a droplet {/Times-New-Roman:Italic=20 r} [{/Symbol m}m]" rotate by 0 at screen 0.27,0.05
# set label 2 " Normalized intensity " rotate by 90 at screen 0.03,0.35
# # set ylabel "Normalized intensity"
# unset ylabel
# unset xlabel
# set xrange[0:1000]
# plot "lineprofile.txt" using ($0*10):1 with linespoints title "Numerical hologram"
# plot "theory.txt" using ($0/10):(sqrt($1)) with lines title "Theoretical intensity"

# unset multiplot

set terminal pdfcairo enhanced font 'Times New Roman,15'
# set size ratio 0.3
set output "lineprofile.pdf"
set key right top box
set margins 0,0,0,0
set xrange[0:1000]
set yrange[0.94:1.06]
unset xlabel
unset ylabel
set multiplot
set label 1 "Distance from the center of a droplet {/Times-New-Roman:Italic=15 r} [{/Symbol=13 m}m]" rotate by 0 at screen 0.27,0.05
set label 2 " Normalized intensity " rotate by 90 at screen 0.03,0.35

set origin 0.15,0.15
set size 0.8,0.35
plot "theory.txt" using ($0/10):(sqrt($1)) with lines title "Theoretical intensity"
set origin 0.15,0.6
set size 0.8,0.35
plot "lineprofile.txt" using ($0*10):1 with linespoints title "Numerical hologram"


unset multiplot
unset output