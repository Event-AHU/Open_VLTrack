%%
clc; clear all; close all; warning off; 
path = '/media/wangxiao/44907FD2907FC946/dataset/TNL2K_dataset/TNL2KJE/';
files =dir(path);
files = files(3:end);

for vid =1:size(files, 1)
    vid
    videoName = files(vid).name;
    imgFiles = dir([path videoName '/imgs/']);
    imgFiles = imgFiles(3:end);
    
    for imgID =1:size(imgFiles, 1)
        imageName = imgFiles(imgID).name; 
        
        try
            image = imread([path videoName '/imgs/' imageName]);
        catch
            disp(['==>> bad case: ', [videoName '/imgs/' imageName]]); 
        end
        
        
    end
    
end

