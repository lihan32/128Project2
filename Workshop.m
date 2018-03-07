%this script will use all function we define to train our network.
%all parameters and input will be set up here.

nneuron=784;
nlayer=5;

W={0.001*rand(784,784),0.001*rand(784),0.001*rand(784),0.001*rand(784,784),0.001*rand(784,784)};
err=zeros(980,1);
for i=1:98
output=Network(i,(train1/256),nlayer,nneuron,W);
error=((double(test1(i,:))/255)-output{nlayer}');
err(i)=norm(error);
delta={};
rate = 0.05;
for j=1:nlayer
    if j==1
        delta=[delta,(output{nlayer}.*(1-output{nlayer}))'.*error];
        W{nlayer}=W{nlayer}+rate*(output{nlayer}.*((1-output{nlayer})'.*error)'*output{nlayer-1}');
    else
    if j~=nlayer
        delta=[delta,delta{j-1}*W{nlayer-j+1}'.*(output{nlayer-j+1}'.*(1-output{nlayer-j+1}'))];
        a=rate*delta{j}'*output{nlayer-j}';
        W{nlayer-j+1}=W{nlayer-j+1}+a;
    else
        delta=[delta,delta{j-1}*W{nlayer-j+1}'.*(output{nlayer-j+1}'.*(1-output{nlayer-j+1}'))];
        a=rate*delta{j}'*double(train0(i,:));
        W{nlayer-j+1}=W{nlayer-j+1}+a;
    end
    end
end
end

