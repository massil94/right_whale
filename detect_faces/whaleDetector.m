images = imageSet('../data/imgs'); % load images .jpg

formatStr = 'negativeFolder/neg%d.jpg'; % output format for negatives
for i=1:images.Count
    imginp = read(images,i); % read an image
    imcropped = imcrop(imginp,[1 1 1078 670]); % Crop
    fileName = sprintf(formatStr,i);
    imwrite(imcropped,fileName); % Save negative images
end