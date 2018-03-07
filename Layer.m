function [ Out ] = Layer( In, n, W )
    %n is the size of the number of neurons in this
    %layor as well as the size of Output data.
    %In is a m*1 vector representing the input data.
    %W is a n*m matrix where the ith row of W is the weights vector of the
    %ith neutrons.
    %Out is a n*1 vector representing the output data.
    
    Out = zeros(n,1);               %initialize Out
    
    for i = 1:n
        [~,OUT]=Neuron(In,W(i,:));  %the ith componant of Out should be
        Out(i,1)=OUT;               %determined by the ith Neuron.
                                    %so here we use an iteration to set up
                                    %all componant of Out.
    end
end

