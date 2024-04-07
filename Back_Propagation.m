train_data = csvread('train_dataset.csv');
test_data = csvread('test_dataset.csv');
X_train = train_data(:, 1:end-1);
y_train = train_data(:, end);
X_test = test_data(:, 1:end-1);
y_test = test_data(:, end);

inputSize = 2;
hiddenSizes = 25;
outputSize = 1;
learningRate = 0.005;
epochs = 200;
momentum = 0.9;
%initialize all weights to the small value
W = cell(1, length(hiddenSizes) + 1);
for i = 1:length(W)
    if i < length(W)
        W{i} = randn(hiddenSizes(i), size(X_train, 2) + 1) * 0.01;
    else
        W{i} = randn(1, hiddenSizes(end) + 1) * 0.01;
    end
end
momentumW = cell(size(W));
for i = 1:length(W)
    if i == 1
        prevLayerSize = inputSize;
    else
        prevLayerSize = hiddenSizes(i-1);
    end
    
    if i == length(W)
        currentLayerSize = outputSize;
    else
        currentLayerSize = hiddenSizes(i);
    end
    
    W{i} = (rand(currentLayerSize, prevLayerSize + 1) * 2 - 1) .* sqrt(1 / (prevLayerSize + 1));
    momentumW{i} = zeros(size(W{i}));
end
trainAccuracies = zeros(epochs, 1);
testAccuracies = zeros(epochs, 1);
trainLosses = zeros(epochs, 1);
testLosses = zeros(epochs, 1);
% Training
for epoch = 1:epochs
    activations = cell(1, length(W) + 1);
    activations{1} = [ones(size(X_train, 1), 1), X_train];
    for i = 1:length(W)
        z = activations{i} * W{i}';
        if i < length(W)
            a = 1 ./ (1 + exp(-z));
            activations{i + 1} = [ones(size(a, 1), 1), a];
        else
            activations{i + 1} = 1 ./ (1 + exp(-z));
        end
    end
    output = activations{end};
    error = y_train - output;
    deltas = cell(1, length(W));
    for i = length(W):-1:1
        if i == length(W)
            deltas{i} = error .* output .* (1 - output);
        else
            deltas{i} = (deltas{i + 1} * W{i + 1}(:, 2:end)) .* activations{i + 1}(:, 2:end) .* (1 - activations{i + 1}(:, 2:end));  % 隐藏层的delta
        end
        
        grad = deltas{i}' * activations{i};
        momentumW{i} = momentum * momentumW{i} + learningRate * grad;
        W{i} = W{i} + momentumW{i};
    end 
    trainPredictions = output > 0.5;
    trainAccuracy = mean(trainPredictions == y_train);
    trainAccuracies(epoch) = trainAccuracy;
    trainLoss = -mean(y_train .* log(output + 1e-9) + (1 - y_train) .* log(1 - output + 1e-9));
    trainLosses(epoch) = trainLoss;
    activationsTest = [ones(size(X_test, 1), 1), X_test];
    for i = 1:length(W)
        zTest = activationsTest * W{i}';
        if i < length(W)
            activationsTest = [ones(size(zTest, 1), 1), 1 ./ (1 + exp(-zTest))];
        else
            testOutput = 1 ./ (1 + exp(-zTest));
        end
    end
    testPredictions = testOutput > 0.5;
    testAccuracy = mean(testPredictions == y_test);
    testAccuracies(epoch) = testAccuracy;
    testLoss = -mean(y_test .* log(testOutput + 1e-9) + (1 - y_test) .* log(1 - testOutput + 1e-9));
    testLosses(epoch) = testLoss;
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Train Accuracy: %.2f%%, Test Accuracy: %.2f%%\n', epoch, trainAccuracy * 100, testAccuracy * 100);
    end
end

figure;
plot(1:epochs, trainAccuracies, 'p-', 'LineWidth', 2);
hold on;
plot(1:epochs, testAccuracies, 'y-', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Accuracy');
title('Training and Test Accuracy over Epochs');
legend('Training Accuracy', 'Test Accuracy');
grid on;
figure;
hold on;
scatter(X_test(y_test == 1, 1), X_test(y_test == 1, 2), 'r', 'filled', 'DisplayName', 'Class 1 - Actual');
scatter(X_test(y_test == 0, 1), X_test(y_test == 0, 2), 'b', 'filled', 'DisplayName', 'Class 0 - Actual');
scatter(X_test(testPredictions ~= y_test, 1), X_test(testPredictions ~= y_test, 2), 100, 'k', 'x', 'DisplayName', 'Misclassified');
title('Test Set Result');
xlabel('X');
ylabel('Y');
legend('show');
grid on;
hold off;
testPredictions_double = double(testPredictions);
C = confusionmat(y_test, testPredictions_double);
figure;
confusionchart(C, {'Class C2', 'Class C1'});
title('Confusion Matrix');
figure;
plot(1:epochs, trainLosses, 'b-', 'LineWidth', 2);
hold on;
plot(1:epochs, testLosses, 'r-', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Binary Cross-Entropy Loss');
title('Training and Test Loss over Epochs');
legend('Training Loss', 'Test Loss');
grid on;
