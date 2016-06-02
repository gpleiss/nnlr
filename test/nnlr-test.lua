require 'nn'
require 'nnlr'

local NnlrTest = torch.TestSuite()
local tester = torch.Tester()

function NnlrTest:assignsLearningRatesAndWeightDecay()
  local module = nn.Linear(2, 4)
    :learningRate('weight', 0.1)
    :learningRate('bias', 0.2)
    :weightDecay('weight', 1)
    :weightDecay('bias', 0)

  local learningRates, weightDecays = module:getOptimConfig(0.1, 0.0001)
  tester:assertTensorEq(
    learningRates,
    torch.Tensor({.1, .1, .1, .1, .1, .1, .1, .1, .2, .2, .2, .2}):typeAs(learningRates) * 0.1,
    1e-5
  )
  tester:assertTensorEq(
    weightDecays,
    torch.Tensor({1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}):typeAs(weightDecays) * 0.0001,
    1e-5
  )
end

function NnlrTest:assignsDefaultValuesWhenNoValueIsGiven()
  local module = nn.Linear(2, 4)
    :learningRate('weight', 0.1)
    :weightDecay('bias', 0)

  local learningRates, weightDecays = module:getOptimConfig(0.1, 0.0001)
  tester:assertTensorEq(
    learningRates,
    torch.Tensor({.1, .1, .1, .1, .1, .1, .1, .1, 1, 1, 1, 1}):typeAs(learningRates) * 0.1,
    1e-5
  )
  tester:assertTensorEq(
    weightDecays,
    torch.Tensor({1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}):typeAs(weightDecays) * 0.0001,
    1e-5
  )
end

-- Edge cases
function NnlrTest:assignsOnlyBiasWhenModuleHasNoWeight()
  local makeModule = function()
    local m = nn.Module()
    m.bias = torch.Tensor(5)
    return m
  end

  tester:assertError(function()
    makeModule():learningRate('weight', 0.1)
  end)

  tester:assertError(function()
    makeModule():weightDecay('weight', 0.1)
  end)

  local module = makeModule()
    :learningRate('bias', 0.2)
    :weightDecay('bias', 0)

  local learningRates, weightDecays = module:getOptimConfig(0.1, 0.0001)
  tester:assertTensorEq(
    learningRates,
    torch.Tensor({.2, .2, .2, .2, .2}):typeAs(learningRates) * 0.1,
    1e-5
  )
  tester:assertTensorEq(
    weightDecays,
    torch.Tensor({0, 0, 0, 0, 0}):typeAs(weightDecays),
    1e-5
  )
end

function NnlrTest:assignsOnlyWeightWhenModuleHasNoBias()
  local makeModule = function()
    local m = nn.Module()
    m.weight = torch.Tensor(4)
    return m
  end

  tester:assertError(function()
    makeModule():learningRate('bias', 0.1)
  end)

  tester:assertError(function()
    makeModule():weightDecay('bias', 0.1)
  end)

  local module = makeModule()
    :learningRate('weight', 0.1)
    :weightDecay('weight', 1)

  local learningRates, weightDecays = module:getOptimConfig(0.1, 0.0001)
  tester:assertTensorEq(
    learningRates,
    torch.Tensor({.1, .1, .1, .1}):typeAs(learningRates) * 0.1,
    1e-5
  )
  tester:assertTensorEq(
    weightDecays,
    torch.Tensor({1, 1, 1, 1}):typeAs(weightDecays) * 0.0001,
    1e-5
  )
end

function NnlrTest:doesNothingForModuleWithNoLearnableParameters()
  local makeModule = function()
    return nn.ReLU()
  end

  tester:assertError(function()
    makeModule():learningRate('weight', 0.1)
  end)

  tester:assertError(function()
    makeModule():weightDecay('weight', 0.1)
  end)

  tester:assertError(function()
    makeModule():learningRate('bias', 0.1)
  end)

  tester:assertError(function()
    makeModule():weightDecay('bias', 0.1)
  end)

  local module = makeModule()

  local learningRates, weightDecays = module:getOptimConfig(0.1, 0.0001)
  assert(not learningRates:storage())
  assert(not weightDecays:storage())
end

-- Test errors
function NnlrTest:onlyAllowsAssigningWeightOrBias()
  local makeModule = function()
    return nn.Linear(3, 4)
  end

  tester:assertError(function()
    makeModule():learningRate('foobar', 0.1)
  end)

  tester:assertError(function()
    makeModule():weightDecay('foobar', 0.1)
  end)
end

-- Containers
function NnlrTest:flattensConfigsOfContainers()
  local makeModule = function()
    local s = nn.Sequential()
    s:add(nn.Linear(2, 2)
      :learningRate('weight', 0.1)
      :learningRate('bias', 0.2)
      :weightDecay('bias', 0)
    )
    s:add(nn.Linear(2, 1)
      :weightDecay('bias', 0)
    )
    return s
  end

  local module = makeModule()
  local learningRates, weightDecays = module:getOptimConfig(0.1, 0.0001)

  tester:assertTensorEq(
    learningRates,
    torch.Tensor({.1, .1, .1, .1, .2, .2, 1, 1, 1}):typeAs(learningRates) * 0.1,
    1e-5
  )
  tester:assertTensorEq(
    weightDecays,
    torch.Tensor({1, 1, 1, 1, 0, 0, 1, 1, 0}):typeAs(learningRates) * 0.0001,
    1e-5
  )
end

tester:add(NnlrTest)
tester:run()
