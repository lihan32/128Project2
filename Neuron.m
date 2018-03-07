function [ NET, OUT ] = Neuron( O,W )
                        %input Matrix O insist of m*n input value;
O=double(O);            %input Matrix W insist of m*n weight value;
NET = sum(dot(W,O));    %NET is the sum of weighted input value,
                        %So, it is the sum of dot product of O and W;
OUT = 1/(1+exp(-NET));  %OUT is the F(NET), and F is provided.

end

