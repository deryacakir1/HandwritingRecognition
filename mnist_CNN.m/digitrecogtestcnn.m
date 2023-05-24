%MATLAB Program to test CNN for Digit Recognition
%Read image for classification
load('handwriting.mat');
[filename,pathname]=uigetfile('*.*','Select the Input Grayscale Image');
filewithpath=strcat(pathname,filename);
I=imread(filewithpath);
I = I(:,:,1);
figure
imshow(I)
%Classify the image using the network
label=classify(net, I);
title(['Recognized Digit is ' char(label)])


