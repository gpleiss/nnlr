require 'nn'
local m = require 'moses'

-----
-- :learningRate() - set relative learning rate for a module
-- Module must have parameters (weight and/or bias)
-----
nn.Module.learningRate = function(self, accessor, value)
  assert(type(accessor) == 'string', 'accessor must be a string')
  assert(type(value) == 'number', 'value must be a number')

  if accessor == 'weight' then
    if self.weight then
      self.__weightLearningRate = value
    else
      error('Tried to assign \'weight\' learningRate, but module has no weight')
    end
  elseif accessor == 'bias' then
    if self.bias then
      self.__biasLearningRate = value
    else
      error('Tried to assign \'bias\' learningRate, but module has no bias')
    end
  else
    error('Unknown accessor type (should be \'weight\' or \'bias\')')
  end
  return self
end

-----
-- :weightDecay() - set relative weight decay for a module
-- Module must have parameters (weight and/or bias)
-----
nn.Module.weightDecay = function(self, accessor, value)
  if accessor == 'weight' then
    if self.weight then
      self.__weightWeightDecay = value
    else
      error('Tried to assign \'weight\' weightDecay, but module has no weight')
    end
  elseif accessor == 'bias' then
    if self.bias then
      self.__biasWeightDecay = value
    else
      error('Tried to assign \'bias\' weightDecay, but module has no bias')
    end
  else
    error('Unknown accessor type (should be \'weight\' or \'bias\')')
  end
  return self
end

-----
-- :optimConfig() -- similar to :parameters(),
-- but for learningRates and weightDecays
-----
nn.Module.optimConfig = function(self, baseLearningRate, baseWeightDecay)
  --local weightLearningRates = self.weight and self.weight:clone():fill((self.__weightLearningRate or 1) * baseLearningRate)
  --local biasLearningRates   = self.bias   and   self.bias:clone():fill((self.__biasLearningRate   or 1) * baseLearningRate)
  local weightWeightDecays  = self.weight and self.weight:clone():fill((self.__weightWeightDecay  or 1) * baseWeightDecay)
  local biasWeightDecays    = self.bias   and   self.bias:clone():fill((self.__biasWeightDecay    or 1) * baseWeightDecay)
  return m.compact({weightLearningRates, biasLearningRates}), m.compact({weightWeightDecays, biasWeightDecays})
end

-- Seperate :optimConfig() method for containers
nn.Container.optimConfig = function(self, baseLearningRate, baseWeightDecay)
  local learningRates = {}
  local weightDecays = {}
  for i, module in ipairs(self.modules) do
    local moduleLearningRates, moduleWeightDecays = module:optimConfig(baseLearningRate, baseWeightDecay)
    table.insert(learningRates, moduleLearningRates)
    table.insert(weightDecays, moduleWeightDecays)
  end
  return m.compact(m.flatten(learningRates)), m.compact(m.flatten(weightDecays))
end

-----
-- :getOptimConfig() -- similar to :getParameters()
-- but for learningRates and weightDecays
-----
nn.Module.getOptimConfig = function(self, baseLearningRate, baseWeightDecay)
  local learningRates, weightDecays = self:optimConfig(baseLearningRate, baseWeightDecay)
  return nn.Module.flatten(learningRates), nn.Module.flatten(weightDecays)
end
