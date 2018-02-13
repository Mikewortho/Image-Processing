train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

% 2. remove the excess non-'4' images
[train_images, train_labels] = nc_equaliseDistribution(4,train_images,train_labels);

% 3. threshold the images
thres_train_images = nc_ohtsuThreshold(train_images);

% 4. get the area and perimeter from the chain code
offsets = [0 1 2 3 4 0 0 0 0; 0 0 0 0 0 1 2 3 4];
quant = 2;
train_params = nc_cooccurParams(thres_train_images,offsets,quant);

%5. Apply Guassian to smooth mage
train_params_g = imgaussfilt(train_params,2);

% 6. measure the variance of the different pixels and discard those which
% are zero
train_stds = std(train_params_g');
tokeep = find(train_stds>0);
train_params_g = train_params_g(tokeep,:);

saveMNIST_csvfile('train_data.csv',train_params_g,train_labels);

% 5. now repeat that for the testing data - don't fiddle the excess
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
thres_test_images = nc_ohtsuThreshold(test_images);
test_params = nc_cooccurParams(thres_test_images,offsets,quant);
test_params_g = imgaussfilt(test_params,2);
test_params_g = test_params_g(tokeep,:);
%5. Apply Guassian to blur and smooth image

saveMNIST_csvfile('test_data.csv',test_params_g,test_labels);

% 6. combine the files
combine_csvfiles('train_data.csv','test_data.csv','train_test_data.csv');
