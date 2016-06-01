package = 'nnlr'
version = '0.0.1-1'

description = {
  summary = 'nnlr',
  detailed = [[
    Add layer-wise learning rate schemes to Torch
  ]]
}

source = {
  url = 'git://github.com/gpleiss/nnlr',
  tag = 'v0.0.1-1',
}

dependencies = {
  'torch >= 7.0',
  'lua ~> 5.1'
}
build = {
  type = 'builtin',
  modules = {
    nnlr = 'nnlr.lua'
  }
}
