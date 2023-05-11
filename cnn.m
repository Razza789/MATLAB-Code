% Load and preprocess the data
data = readtable('movie.csv');

% Shuffle the data
data = data(randperm(size(data, 1)), :);

reviews = data.text;
cleanedReviews = cleanText(reviews);
documents = tokenizedDocument(cleanedReviews);

% Encode the sentiment labels as categorical
labels = categorical(data.label);

% Split the data into train (80%) and test (20%) sets
trainRatio = 0.8;
trainCount = floor(trainRatio * size(documents, 1));

trainDocuments = documents(1:trainCount);
testDocuments = documents(trainCount + 1:end);

trainLabels = labels(1:trainCount);
testLabels = labels(trainCount + 1:end);

% Train the word embedding
embeddingDimension = 100;
embedding = trainWordEmbedding(trainDocuments, 'Dimension', embeddingDimension, 'NumEpochs', 5);

% Convert the documents to sequences
sequenceLength = 50;
XTrain = doc2sequence(embedding, trainDocuments, 'Length', sequenceLength);
XTest = doc2sequence(embedding, testDocuments, 'Length', sequenceLength);
% Calculate the shortest sequence length in the training data
minSequenceLength = min(cellfun(@length, XTrain));

% Define the CNN network architecture
numClasses = 2;
numFeatures = embeddingDimension;
numResponses = numClasses;
numHiddenUnits = 100;
numFilters = 128;
filterSize = 5;

layers = [ ...
    sequenceInputLayer(numFeatures)
    convolution1dLayer(filterSize, numFilters, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    globalMaxPooling1dLayer
    fullyConnectedLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    softmaxLayer
    classificationLayer];

% Network training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(XTrain, trainLabels, layers, options);

% Test the trained model
predictions = classify(net, XTest);

% Calculate the accuracy
totalPredictions = numel(testLabels);
correctPredictions = sum(predictions == testLabels);
accuracy = correctPredictions / totalPredictions;

% Display the results
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Calculate the confusion matrix
cm = confusionmat(testLabels, predictions);

% Create category labels
categories = {'True Negative', 'False Positive', 'False Negative', 'True Positive'};

% Flatten the confusion matrix and create a bar chart
figure;
bar(reshape(cm, [1, numel(cm)]));
set(gca, 'XTickLabel', categories);

% Set the title and axis labels
title('Test Results');
xlabel('Category');
ylabel('Number of Instances');