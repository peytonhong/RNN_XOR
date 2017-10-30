function layer = sigmoid(logit, layer_size)
layer = zeros(layer_size, 1);
for i=1:layer_size
    layer(i) = 1/(1+exp(-logit(i)));
end
end