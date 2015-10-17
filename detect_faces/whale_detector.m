% Use Training Image Labeler
% Open session 'labelingSession.mat'
% Export ROIs (so that positiveInstance can be in the workspace)

trainCascadeObjectDetector('detectorFile.xml', positiveInstances, 'negativeFolder','FalseAlarmRate',0.01,'NumCascadeStages',15,'FeatureType','LBP');
detector = vision.CascadeObjectDetector('detectorFile.xml');
img = imread('../data/imgs/w_915.jpg');
bbox = step(detector,img);
detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'whale face');
figure;
imshow(detectedImg);