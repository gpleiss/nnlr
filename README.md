# nnlr

Add layer-wise learning rate schemes to Torch.
At the moment, it works with `nn` and `nngraph` modules.
At the moment, the only supported optimization algorithm
supported is [optim](https://github.com/torch/optim)
SGD implementation.

## Usage

`nnlr` adds the following methods to `nn.Module`:

```lua
module:learningRate('weight', 0.1)
module:learningRate('bias', 0.2)
module:weightDecay('weight', 1)
module:weightDecay('bias', 0)
```

The `learningRate` and `weightDecay` methods set the
`module`'s **relative** learning rate and weight decay, respectivly.
I.e., if the learning rate for the network is 0.05, then the
weight learning rate of `module` will be 0.005, and the bias learning
rate 0.01.

All of these methods are optional. If the relative learning rate or weight
decay is not set for a module, it will default to 1. Additionally, each
method returns the original module, allowing for chaining.

Rather than suppling a scalar learning rate and weight decay to the
optimization function, supply the following vectors:

```lua
local learningRates, weightDecays = module:getOptimConfig(baseLearningRate, baseWeightDecay)
```

The SGD config table should then be of the form:

```lua
{
  learningRates = learningRates,
  weightDecays = weightDecays,
  learningRate = baseLearningRate,
  -- ...
}
```
Note that the config table uses the keys `learningRates` and `weightDecays` (**plural**).

(The API is inspired by the [nninit](https://github.com/Kaixhin/nninit) package.
These two packages should work well in conjunction.)


## Installation

 ```sh
luarocks install nnlr
```

## Example

```lua
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
net:add(nn.SpatialBatchNormalization(32))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-------
-- This layer has a lower learning rate than all the
-- other layers.
-------
-- Conv 2
net:add(nn.SpatialConvolution(32, 48, 5, 5, 1, 1, 1, 1)
  :learningRate('weight', 0.1)
  :learningRate('bias', 0.2)
  -- we don't supply a weightDecay value for 'weight' --- rather we
  -- choose to use the default value
  :weightDecay('bias', 0)
)
net:add(nn.SpatialBatchNormalization(48))
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
net:add(nn.BatchNormalization(100))
net:add(nn.ReLU())
-- Full 4
net:add(nn.Linear(100, 100))
net:add(nn.BatchNormalization(100))
net:add(nn.ReLU())
-- Full 5
net:add(nn.Linear(100, 10))
net:add(nn.LogSoftMax())

-------
-- Here we get the learningRates and weightDecays
-- vectors required for optimization
-------
local baseLearningRate = 0.1
local baseWeightDecay = 0.0001
local learningRates, weightDecays = net:getOptimConfig(baseLearningRate, baseWeightDecay)

-------
-- Train the network...
-------

local weight, grad = net:getParameters()

-- ... some training loop ...
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
    learningRate = baseLearningRate,
    momentum = 0.9,
  })

-- ...

