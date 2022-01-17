from subprocess import call
import glob
import cv2

paths = glob.glob("./*.png")
imageNames = []
for ii in range(len(paths)):
    tmp = paths[ii].replace("./","").replace(".png","")
    imageNames.append(tmp)

for ii in range(len(paths)):
    im = cv2.imread(paths[ii])
    size = im.shape[0]

    gnuplotCommand = f'''
    set terminal pdfcairo enhanced font 'Times New Roman,15';
    unset key;
    set size ratio 1;
    set xlabel '{{/Times-New-Roman:Italic=20 x}} [pixel]';
    set ylabel '{{/Times-New-Roman:Italic=20 y}} [pixel]';
    set xrange[0:{size}];
    set yrange[0:{size}];
    set tics out;
    set output "./pdfprocessed/{imageNames[ii]}.pdf";
    plot "./{imageNames[ii]}.png" binary filetype=png origin=(0.5,0.5) with rgbimage;
    unset output;
    '''

    call( [ "gnuplot", "-e", gnuplotCommand])

    print(f"Output seccessed {paths[ii]}\n")