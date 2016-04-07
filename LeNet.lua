require 'nn'
require 'paths'
require 'gnuplot'

-- download cifar-10 dataset if not already exists
if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

print("Finish loading datasets")

-- load datasets and labels
trainset = torch.load('cifar10torchsmall/cifar10-train.t7')
testset = torch.load('cifar10torchsmall/cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

-- process trainset
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
testset.data = testset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

-- normalize the input data
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future

for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- define the LeNet
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())   

criterion = nn.ClassNLLCriterion() -- set the loss function

-- define a one-epoch trainer for iterating
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 1 -- just do 1 epoch of training for each iteration

-- build function for test accuracy
function accuracy(net)
	correct = 0
	for i=1,100 do
		local groundtruth = testset.label[i]
	    local prediction = net:forward(testset.data[i])
	    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	    if groundtruth == indices[1] then
	        correct = correct + 1
	    end
	end
	return correct
end    

-- get a small sample 
train_small = trainset[{ {1, 1000} }]
train_small["data"] = train_small[1]
train_small["label"] = train_small[2]
train_small[1] = nil
train_small[2] = nil


setmetatable(train_small, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

function train_small:size() 
    return self.data:size(1) 
end

-- declare an table variable to record the correctness over time
a = {}
iter = {}

-- loops through the training epochs times
for epoch=1,30 do
	print("Epoch" .. epoch)

	trainer:train(train_small)
	local cor = accuracy(net)

	print(cor)

	table.insert(iter, epoch)
	table.insert(a, cor)
end

-- plot the overall accuracy
gnuplot.figure(1)
gnuplot.title('Test accuracy over iterations')
gnuplot.plot(torch.Tensor(iter), torch.Tensor(a))


