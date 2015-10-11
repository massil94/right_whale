images = imageSet('../right_whale_hunt/imgs'); % load images .jpg

formatStr = 'neg%d.jpg'; % output format for negatives
for i=1:images.Count
    imginp = read(images,i); % read an image
    using imcropped = imcrop(imginp,[1 1 1078 670]); % Crop
    fileName = sprintf(formatStr,i);
    imwrite(imcropped,fileName); % Save negative images
end

