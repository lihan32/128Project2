function [ output ] = Network( n,Train,nlayer, nneuron,W)
    %n is the parameter for Read function that extract nth row of Train.
    %Train is the input training set (a matrix).
    %nlayer is the number of hidden layer and output layer
    %nneuron is the number of neurons in each hidden layer
    %W is a collection of matrix that
    %represent the weighted matrix bewteen each layer.
    %output is the collection that record all ouput from each layer
    
    output={};
    input = Read(Train,n,0);        %input value is nth row of train matrix
    [~,m]=size(input);                %set m be the size of input
    %hidden layer
    for i=1:nlayer-1
        input=Layer(input,nneuron,W{1,i}); %to other layers, we need to use the weight matrix W
        output=[output,input];
    end
    %output layer
    input=Layer(input,10,W{1,nlayer});%finally we classify it to 0-9
    output=[output,input];
end

