%this script will use all function we define to train our network.
%all parameters and input will be set up here.

nneuron=10;%num of neuron in a layer
nlayer=5;%num of layer
%generate a random matrix
R={0.01*rand(10,784),0.01*rand(10,10),0.01*rand(10,10),0.01*rand(10,10),0.01*rand(10,10)};
W=R;
err=zeros(980,1);%measure the norm of error bewteen result and target

%here I write a section to do the reverse propagation for a traing set.
%If we need to train it in more than one set.
%we need to store the weight matrixes W manually and run for next training
%set.

%reverse propagation here
for i=1:1980%determine the number of training pairs here
output=Network(i,(double(train1)),nlayer,nneuron,W);%specify the training set and call for network
save=double(train1);
error=abs([0,1,0,0,0,0,0,0,0,0]-output{nlayer}');%specify the target and calculate error
err(i)=norm(error);
delta={};
rate = 0.05;%this is the arbitrary in the document




    for CurrentLayer=nlayer:-1:1
        if CurrentLayer==nlayer
            for p=1:10
                Tdelta=zeros(10,1);%delta for current layer
                %this loop construct the delta for current layer
                for q=1:10
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
                for p=1:10
                    Tdelta=zeros(10,1);
                    for q=1:10
                        o=output{CurrentLayer}(q);
                        %the formula of delta is diff from the output layer
                        Tdelta(q)=o*(1-o)*delta{nlayer-CurrentLayer}'*W{CurrentLayer+1}(:,p);
                    end
                    W{CurrentLayer}(:,p)=W{CurrentLayer}(:,p)+rate*Tdelta*output{CurrentLayer-1}(p);
                end
                delta=[delta,Tdelta];
            else
                for p=1:10
                    Tdelta=zeros(10,1);
                    for q=1:10
                        o=output{CurrentLayer}(q);
                        Tdelta(q)=o*(1-o)*delta{nlayer-CurrentLayer}'*W{CurrentLayer+1}(:,p);
                    end
                    %the previous output for the first hidden layer is the
                    %input layer
                    W{CurrentLayer}(:,p)=W{CurrentLayer}(:,p)+rate*Tdelta*save(i,p);
                end
                delta=[delta,Tdelta];
            end
        end
    end
end


