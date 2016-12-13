
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



print('<trainer> reloading previously trained network')
model = torch.load(opt.network)

model = model:double()

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

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



classes = {'1','2','3','4','5','6','7','8','9','10'}

confusion = optim.ConfusionMatrix(classes)

local preds = model:forward(inputs)

for i = 1,opt.batchSize do
    confusion:add(preds[i], targets[i])
end

print(confusion)
confusion:zero()

inputs_adv = adversarial_fast(model, criterion, inputs, targets, 1, 0.3)

local preds = model:forward(inputs_adv)

for i = 1,opt.batchSize do
    confusion:add(preds[i], targets[i])
end

print(confusion)
-- print(inputs_adv)