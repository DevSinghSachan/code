load_imagenet_model();
predictions = test("data/doggy.png");
fprintf('Top-5 predictions:\n');
predictions(1:5)
