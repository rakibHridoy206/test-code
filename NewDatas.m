clc, clear;
data = xlsread('ROWC.xlsx')
figure
plot(data)
xlabel("Days")
ylabel("Closing Prices")
title("Daily closing price of Brak Bank")

%Train and Test Data
numTimeStepsTrain = floor(0.9*numel(data));
x = (numTimeStepsTrain+1)/2;
train = data(1:numTimeStepsTrain+1)
test = data(numTimeStepsTrain+1:end)


%Standardize Data
mu=mean(train);
sigma=std(train);
std_train_data = (train-mu)/sigma;
XTrain = std_train_data(1:end-1);
YTrain = std_train_data(2:end);

%Define LSTM Network
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;
layers = [sequenceInputLayer(numFeatures),lstmLayer(numHiddenUnits),fullyConnectedLayer(numResponses),regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress'); 
net = trainNetwork(XTrain,YTrain,layers,options);

%Forecast Future TimeSteps
std_test_data = (test-mu)/sigma;
XTest = std_test_data(1:end-1);
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
num_steps_test = numel(XTest);
for i=2:num_steps_test
    [net,YPred(:,i)] = predictAndUpdateState(net,YTrain(:,i-1));
end
YPred = YPred*sigma+mu;
YTest = test(2:end);
rmse = sqrt(mean((YPred-YTest).^2));
figure
plot(train(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+num_steps_test);
plot(idx,[data(numTimeStepsTrain), YPred], '.-')
legend(["Observed" "Forecast"])
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred, '.-')
hold off
legend(["Observed" "Forecast"])
subplot(2,1,2)
stem(YPred-YTest)
ylabel("Error")
title("RMSE: "+rmse)

%Update Network State with Observed Values
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
YPred = [];
for i=1:num_steps_test
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i));
end
YPred = sigma*YPred+mu;
rmse = sqrt(mean((YPred-YTest).^2));
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred, '.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Closing Prices")
xlabel("Days")
title("Forecast with updates")
subplot(2,1,2)
stem(YPred-YTest)
ylabel("Error")
xlabel("Days")
title("RMSE = " + rmse)








