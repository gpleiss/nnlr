require 'nn'
require 'optim'
require 'nnlr'

-- Network
local net = nn.Sequential()

-------
-- This layer is locked down. No learning happens
-------
-- Conv 1
net:add(nn.SpatialConvolution(1, 32, 5, 5, 1, 1, 2, 2)
  :learningRate('weight', 0)
  :learningRate('bias', 0)
  :weightDecay('weight', 0)
  :weightDecay('bias', 0)
)
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-------
-- This layer has a lower learning rate than all the
-- other layers
-------
-- Conv 2
net:add(nn.SpatialConvolution(32, 48, 5, 5, 1, 1, 1, 1)
  :learningRate('weight', 0.1)
  :learningRate('bias', 0.2)
  :weightDecay('weight', 1)
  :weightDecay('bias', 0)
)
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.View(-1):setNumInputDims(3))

-------
-- The following layers use the default learning rate
-- and weight decay. No learningRate or weightDecay
-- call necessary.
-------
-- Full 3
net:add(nn.Linear(2352, 100))
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
-- Full 4
net:add(nn.Linear(100, 100))
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
-- Full 5
net:add(nn.Linear(100, 10))
net:add(nn.LogSoftMax())


-------
-- Here we get the learningRates and weightDecays
-- vectors required for optimization
-------
local baseLearningRate = 0.01
local baseWeightDecay = 0.0005
local weight, grad = net:getParameters()
local learningRates, weightDecays = net:getOptimConfig(baseLearningRate, baseWeightDecay)


-------
-- Train the network...
-------

-- Config
local geometry = {32, 32}
local batchSize = 128

-- Dataset
require 'examples.helpers.dataset-mnist'
mnist.download()
local dataset = mnist.loadTrainSet(6400, geometry)
dataset:normalizeGlobal()

-- Criterion + metric
local criterion = nn.ClassNLLCriterion()
local metric = optim.ConfusionMatrix({1, 2, 3, 4, 5, 6, 7, 8, 9, 0})

-- Train
for i = 1, dataset:size(), batchSize do
  local input = torch.Tensor(batchSize, 1, geometry[1], geometry[2])
  local label = torch.Tensor(batchSize)
  for j = 1, batchSize do
    local sample = dataset[i + j - 1]
    input[j]:copy(sample[1])
    _, label[j] = sample[2]:max(1)
  end

  local output = net:forward(input)
  local loss = criterion:forward(output, label)
  local gradOutput = criterion:backward(output, label)
  net:backward(input, gradOutput)

  print(i, loss)
  metric:batchAdd(output, label)

  local feval = function()
    return loss, grad
  end

  -------
  -- We use the learningRates and weightDecays vectors here
  -- in place of scalar values
  -------
  optim.sgd(feval, weight, {
    learningRates = learningRates,
    weightDecays = weightDecays,
    momentum = 0.9,
  })

end

print(metric)
