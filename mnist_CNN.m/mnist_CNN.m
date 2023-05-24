% This code implement the classification of MNIST Data, which has validated
% under Matlab2018b and Matlab2020b. To use it, the MNIST data set can be
% downloaded from http://yann.lecun.com/exdb/mnist/ . For most instance the
% code can get a classification accuracy greater than 99% under the test
% data set. but this situation does not happens in probability 100%.

clear all;
close all;

N_sample = 60000;
N_test=10000;
XTrain = zeros(28,28,1,N_sample); %Başlangıçta sıfırlardan oluşan bir matrix oluşturmak için 'zeros' metodu kullanılır.

YTrain=zeros(N_sample,1); %Etiket işlemidir. Bilgi düzeni için uygulanır. Matrix vektör olarak değerlendirilebilir.

% Please dowload the MNIST data set from http://yann.lecun.com/exdb/mnist/
% and unzip.
fidimg1=fopen('train-images.idx3-ubyte','rb');
fidimg2=fopen('train-labels.idx1-ubyte','rb');

[img,count]=fread(fidimg1,16);   % table head
[imgInd,count1]=fread(fidimg2,8);   %table head
for k=1:N_sample    
    [im,~]=fread(fidimg1,[28,28]);
    ind=fread(fidimg2,1);
    XTrain(:,:,1,k)=im'; %Matrixse atama işlemini gerçekleştiren kod satırıdır. Başlangıçta değerler sıfırdır (zeros metodu), bu metodla değerler atanır.
    YTrain(k)=ind;
end
fclose(fidimg1);
fclose(fidimg2);
YTrain=categorical(YTrain);

XTest = zeros(28,28,1,N_test);
YTest=zeros(N_test,1);
fidimg1=fopen('t10k-images.idx3-ubyte','rb'); %Dosya yükleme ve açma işlemi için yazılmıştır, 'rb' okuma işlemi anlamına gelir.
fidimg2=fopen('t10k-labels.idx1-ubyte','rb');

[img,count]=fread(fidimg1,16);
[imgInd,count1]=fread(fidimg2,8);
for k=1:N_test    
    [im,~]=fread(fidimg1,[28,28]);
    ind=fread(fidimg2,1);
    XTest(:,:,1,k)=im';% training set building
    YTest(k)=ind;
end
fclose(fidimg1);
fclose(fidimg2);
YTest=categorical(YTest); %Bu kod etiketi kategori haline getirir ve veriyi eğitirken yardımcı olur. 

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]); %Daha iyi sonuç alabilmek için veri arttırma işlemidir. Ne kadar çok veri olursa algoritma o kadar iyi sonuç verir. 
imageSize = [28 28 1];
augimds = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);

layers=[imageInputLayer([28 28 1],'Name','input')
        convolution2dLayer(3,6,'Padding','same') %3:filtre sayısıdır, 6:filtre boyutudur
        reluLayer %Aktivasyon fonksiyonudur. Verilerin ne kadarının çıktı olacağına bakar. Layer katman oluşturur.
        batchNormalizationLayer
        maxPooling2dLayer(2,'Stride',2)% 2*2 havuzlama katmanı, Oluşturulan matrixte en büyük (maxPooling) değeri alır, (minPooling) en küçük değeri alır..
        convolution2dLayer(3,16)
        reluLayer
        batchNormalizationLayer
        maxPooling2dLayer(2,'Stride',2)
        fullyConnectedLayer(120,'name','f1')
        reluLayer
        fullyConnectedLayer(84,'name','f2')
        reluLayer
        fullyConnectedLayer(10,'name','f3')
        softmaxLayer
        classificationLayer];
%Daha düşük boyutlu olduğunda, iyi sonuç verir.
options = trainingOptions('adam','MaxEpochs',25,'LearnRateSchedule' ,'piecewise','LearnRateDropPeriod',15,'LearnRateDropFactor' ,0.1);
tic;
net = trainNetwork(augimds,layers,options);
toc;
YPred = classify(net,XTest);
accuracy = sum(YTest==YPred)/numel(YTest);
z=find(YTest~=YPred);
gamma=[28 28];
numStart=1;
numCol=5;
subplot(121)
imOrg=zeros(numCol*gamma(1),5*gamma(2));
for k=1:numCol
    imOrg((k-1)*gamma(1)+1:k*gamma(1),:)=reshape(XTest(:,:,1,z((k-1)*numCol+1:k*numCol)),28,28*numCol);
end
imshow(imOrg);
xx=reshape(double(YPred(z(1:numCol*5))),numCol,5);
title(num2str(xx));

