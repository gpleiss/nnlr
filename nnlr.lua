require 'nn'

nn.Module.learningRate = function(self, accessor, value)
  return self
end

nn.Module.weightDecay = function(self, accessor, value)
  return self
end

nn.Module.getOptimConfig = function(self, baseLearningRate, baseWeightDecay)
  local learningRates = self:getParameters():clone():fill(baseLearningRate)
  local weightDecays = learningRates:clone():fill(baseLearningRate)
  return learningRates, weightDecays
end
