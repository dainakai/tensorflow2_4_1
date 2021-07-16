from subprocess import call
import glob
import cv2

paths = glob.glob("./*.png")
# imageNames = paths.replace("./currentcnn/",'')
imageNames = []
for ii in range(len(paths)):
    tmp = paths[ii].replace("./","")
    imageNames.append(tmp)

for ii in range(len(paths)):
    im = cv2.imread(paths[ii])
    size = im.shape[0]

    gnuplotCommand = f'''
    set terminal pngcairo enhanced font 'Times New Roman,15';
    unset key;
    set size ratio 1;
    set xlabel '{{/Times-New-Roman:Italic=20 x}} [pixel]';
    set ylabel '{{/Times-New-Roman:Italic=20 y}} [pixel]';
    set xrange[0:{size}];
    set yrange[0:{size}];
    set tics out;
    set output "./processed/{imageNames[ii]}";
    plot "./{imageNames[ii]}" binary filetype=png origin=(0.5,0.5) with rgbimage;
    '''

    call( [ "gnuplot", "-e", gnuplotCommand])

    print(f"Output seccessed {paths[ii]}\n")