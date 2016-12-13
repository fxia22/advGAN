
require 'torch'
require 'nn'
require 'cunn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 100)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   --type             (default "float")     use cuda
]]


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end



print('<trainer> reloading previously trained network')
model = torch.load(opt.network)

modelD = model:double()

-- retrieve parameters and gradients

-- verbose
print('<mnist> using model:')
print(modelD)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
criterion = nn.ClassNLLCriterion()

adversarial_fast = dofile('adversarial-fast.lua')


----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)


geometry = {32,32}
dataset = trainData
local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
local targets = torch.Tensor(opt.batchSize)

local k = 1



classes = {'1','2','3','4','5','6','7','8','9','10'}



local intensity = 0.7


local nc = 1
local nz = opt.nz
local ngf = 8
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is (nc) x 64 x 64
netG:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.LeakyReLU(0.2, true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*8) x 4 x 4

-- netG:add(SpatialConvolution(ngf * 8, ngf * 16, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ndf * 16)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*16) x 2 x 2

-- netG:add(SpatialFullConvolution(ngf*16, ngf * 8, 4, 4))
-- netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
netG:add(nn.Clamp(-intensity, intensity))


-- state size: (nc) x 64 x 64

netG:apply(weights_init)

modelG = nn.Sequential()
         :add(nn.ConcatTable()
            :add(netG)
            :add(nn.Identity()))
         :add(nn.CAddTable(true))

print(modelG)


local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
local targets = torch.Tensor(opt.batchSize)

local k = 1

for i = 1,math.min(opt.batchSize,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
end


confusion = optim.ConfusionMatrix(classes)

inputs_adv = modelG:forward(inputs)

local preds = modelD:forward(inputs_adv)

confusion:zero()
for i = 1,opt.batchSize do
    confusion:add(preds[i], targets[i])
end

print(confusion)




require 'optim'


parameters,gradParameters = modelG:getParameters()
optimStateG = {
   learningRate = 0.01,
}


parametersD,gradParametersD = modelD:getParameters()
optimStateD = {
   learningRate = 0.01,
}

disp = require 'display'

function feval(parameters)
      gradParameters:zero()
      local gen = modelG:forward(inputs)
      local outputs = modelD:forward(gen)
      disp.image(gen-inputs , {win=67, title='mnist'})
      disp.image(inputs , {win=69, title='mnist-input'})
      disp.image(gen , {win=68, title='mnist-gen'})
      loss = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      local df_dg = modelD:updateGradInput(gen, df_do):mul(-1)
      modelG:backward(inputs, df_dg)
      -- print('<trainer> loss on training set:'..loss)
      return loss, gradParameters
end


function fevalD(parametersD)
      gradParametersD:zero()
      local gen = modelG:forward(inputs)
      local outputs = modelD:forward(gen)
      
      local df_do = criterion:backward(outputs, targets)

      modelD:backward(gen, df_do)
      -- print('<trainer> loss on training set:'..loss)
      return loss, gradParametersD
end


fd = io.open('temp4.txt', 'w')

for epoch = 1, 100 do
   -- local function we give to optim
   -- it takes current weights as input, and outputs the loss
   -- and the gradient of the loss with respect to the weights
   -- gradParams is calculated implicitly by calling 'backward',
   -- because the model's weight and bias gradient tensors
   -- are simply views onto gradParams

    for t = 1,dataset:size(),opt.batchSize do
           local k = 1
           for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
              -- load new sample
              local sample = dataset[i]
              local input = sample[1]:clone()
              local _,target = sample[2]:clone():max(1)
              target = target:squeeze()
              inputs[k] = input
              targets[k] = target
              k = k + 1
           end
        
           optim.sgd(feval, parameters, optimStateG)
        
           optim.sgd(fevalD, parametersD, optimStateD)
           
           inputs_adv = modelG:forward(inputs)
           
           local preds = modelD:forward(inputs_adv)
           confusion = optim.ConfusionMatrix(classes)

           confusion:zero()
           for i = 1,opt.batchSize do
               confusion:add(preds[i], targets[i])
           end

           print(confusion)
           print(confusion.totalValid * 100)
           fd:write(confusion.totalValid * 100)
           fd:write('\n')

           -- accuracies
           
    end

    

   
end

fd:close()

-- train G
