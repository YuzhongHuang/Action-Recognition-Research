require "nngraph"
require "rnn"

-- I will start with the LeNet I build with NN to visualize my network

-- define the LeNet function

function get_LeNet()
    -- it is common style to mark inputs with identity nodes for clarity.
    local input = nn.Identity()()
    
    -- each hidden layer is achieved by connecting the previous one
    -- here we define a single hidden layer network
    local conv1 = nn.SpatialConvolution(3, 6, 5, 5)(input) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
    local conv1_a = nn.ReLU()(conv1) -- non-linearity 
    local pool1 = nn.SpatialMaxPooling(2,2,2,2)(conv1_a) -- A max-pooling operation that looks at 2x2 windows and finds the max.

    local conv2 = nn.SpatialConvolution(6, 16, 5, 5)(pool1) -- 6 input image channels, 16 output channels, 5x5 convolution kernel
    local conv2_a = nn.ReLU()(conv2) -- non-linearity 
    local pool2 = nn.SpatialMaxPooling(2,2,2,2)(conv2_a) -- A max-pooling operation that looks at 2x2 windows and finds the max.
   
    local reshape = nn.View(16*5*5)(pool2) -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5

    local fullConnect1 = nn.Linear(16*5*5, 120)(reshape) -- fully connected layer (matrix multiplication between input and weights)
    local fullConnect1_a = nn.ReLU()(fullConnect1) -- non-linearity 
    
    local fullConnect2 = nn.Linear(120, 84)(fullConnect1_a)
    local fullConnect2_a = nn.ReLU()(fullConnect2) -- non-linearity 
    
    local fullConnect3 = nn.Linear(84, 10)(fullConnect2_a) -- 10 is the number of outputs of the network (in this case, 10 digits)
    local softMax = nn.LogSoftMax()(fullConnect3)
    
    -- the following function call inspects the local variables in this
    -- function and finds the nodes corresponding to local variables.
    nngraph.annotateNodes()
    return nn.gModule({input}, {softMax})
end
    
LeNet = get_LeNet()

function get_LCRN()
    -- it is common style to mark inputs with identity nodes for clarity.
    local input = nn.Identity()()
    
    -- each hidden layer is achieved by connecting the previous one
    -- here we define a single hidden layer network
    local conv1 = nn.SpatialConvolution(1, 6, 5, 5)(input) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
    local conv1_a = nn.ReLU()(conv1) -- non-linearity 
    local pool1 = nn.SpatialMaxPooling(2,2,2,2)(conv1_a) -- A max-pooling operation that looks at 2x2 windows and finds the max.

    local conv2 = nn.SpatialConvolution(6, 16, 5, 5)(pool1) -- 6 input image channels, 16 output channels, 5x5 convolution kernel
    local conv2_a = nn.ReLU()(conv2) -- non-linearity 
    local pool2 = nn.SpatialMaxPooling(2,2,2,2)(conv2_a) -- A max-pooling operation that looks at 2x2 windows and finds the max.
   
    local reshape = nn.View(16*5*5)(pool2) -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5

    local fullConnect1 = nn.Linear(16*5*5, 120)(reshape) -- fully connected layer (matrix multiplication between input and weights)
    local fullConnect1_a = nn.ReLU()(fullConnect1) -- non-linearity 
    
    local fullConnect2 = nn.Linear(120, 84)(fullConnect1_a)
    local fullConnect2_a = nn.ReLU()(fullConnect2) -- non-linearity 
    
    local fullConnect3 = nn.Linear(84, 10)(fullConnect2_a) -- 10 is the number of outputs of the network (in this case, 10 digits)
    local softMax = nn.LogSoftMax()(fullConnect3)
    
    -- concatnate with LSTM
    local split = nn.SplitTable(1)(softMax)
    local sequence = nn.Sequencer(nn.LSTM(10,10,5))(split)
    local selectTable = nn.SelectTable(-1)(sequence)
    local output = nn.LogSoftMax()(selectTable)
    
    -- the following function call inspects the local variables in this
    -- function and finds the nodes corresponding to local variables.
    nngraph.annotateNodes()
    return nn.gModule({input}, {output})
end

LCRN = get_LCRN()