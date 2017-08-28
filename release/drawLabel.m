figure
load example1.mat
box_color = {'red','green','yellow'};

for i=1:15
    
    im=insertText(im,xy(:,i)',num2str(i),'FontSize',12,'TextColor','white','BoxColor','red','BoxOpacity',0.4);
end
imshow(im)