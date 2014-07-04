load_imagenet_model();
imgs = get_images();
predictions = test(imgs);
for i = 1:128
  imagesc(squeeze(imgs(i, :, :, :)));
  display(predictions(i, 1));
  waitforbuttonpress
end
