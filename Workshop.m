%this script will use all function we define to train our network.
%all parameters and input will be set up here.

nneuron=784;%num of neuron in a layer
nlayer=18;%num of layer
%generate a random matrix
W={0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005,0.001*rand(784,784)-0.005};
err=zeros(980,1);%measure the norm of error bewteen result and target

%reverse propagation here
for i=1:198%determine the number of training pairs here
output=Network(i,(double(train1)/256),nlayer,nneuron,W);%restore output
error=abs((double(test1(i,:))/256)-output{nlayer}');%calculate error
err(i)=norm(error);
delta={};
rate = 0.05;%this is the arbitrary in the document



for CurrentLayer=nlayer:1
    for CurrentLayer=nlayer:-1:1
        if CurrentLayer==nlayer
            for p=1:784
                Tdelta=zeros(784,1);%delta for current layer
                %this loop construct the delta for current layer
                for q=1:784
                    o=output{CurrentLayer}(q);
                    Tdelta(q)=o*(1-o)*error(q);
                end
                %calculate w(n+1) here
                W{CurrentLayer}(:,p)=W{CurrentLayer}(:,p)+rate*Tdelta*output{CurrentLayer-1}(p);
            end
            %restore delta of this layer
            delta=[delta,Tdelta];
        else
            if CurrentLayer~=1
                for p=1:784
                    Tdelta=zeros(784,1);
                    for q=1:784
                        o=output{CurrentLayer}(q);
                        %the formula of delta is diff from the output layer
                        Tdelta(q)=o*(1-o)*delta{nlayer-CurrentLayer}'*W{CurrentLayer+1}(:,p);
                    end
                    W{CurrentLayer}(:,p)=W{CurrentLayer}(:,p)+rate*Tdelta*output{CurrentLayer-1}(p);
                end
                delta=[delta,Tdelta];
            else
                for p=1:784
                    Tdelta=zeros(784,1);
                    for q=1:784
                        o=output{CurrentLayer}(q);
                        Tdelta(q)=o*(1-o)*delta{nlayer-CurrentLayer}'*W{CurrentLayer+1}(:,p);
                    end
                    %the previous output for the first hidden layer is the
                    %input layer
                    W{CurrentLayer}(:,p)=W{CurrentLayer}(:,p)+rate*Tdelta*double(train1(i,p))/256;
                end
                delta=[delta,Tdelta];
            end
        end
    end
end

end

