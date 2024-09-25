clear all;
clc;

rawpath='./dataset/raw/test/clean/';
rgbpath='./dataset/rgb/test/clean/';

lists1=dir(rawpath);
n=length(lists1);

for i=3:n
    raw_filename = strcat(rawpath, lists1(i).name);
    raw = imread(raw_filename);
    raw = demosaic(raw, 'gbrg');
    raw = double(raw) / 65535;

    raw(:,:,1)=raw(:,:,1)*1.82;
    raw(:,:,2)=raw(:,:,2);
    raw(:,:,3)=raw(:,:,3)*1.61;
    
    rgb = lin2rgb(raw);

    rgb_filename = append(rgbpath, lists1(i).name);
    imwrite(rgb, rgb_filename)
end
