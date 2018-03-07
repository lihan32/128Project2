function [ M ] = Read( N,n,index )
                            %N is the input data matrix
                            %n specify which row of N we want to read
                            %index intermined if we want to plot
                            %M return what we want
                                                    
M = N(n,:);                 %extract the nth row of matrix N 

if index                    %if index~= 0, we plot this M
    digit = reshape(M,28,28);   %reshape this row to 28*28 matrix
    digit = rot90(flipud(digit),-1);%do some rotation
    image(digit),
    colormap(gray(256)), axis square tight off;
end
                            %output is M, which is what we want
end

