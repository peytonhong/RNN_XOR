% XOR problem using RNN (Recurrent Neural Network)
% Intelligence Control Midterm exam (지능제어 중간고사)
% 2017.10.24 Hyosung Hong (홍효성)

clear; close all; clc;

num_input = 2;
num_hidden = 20;
num_output = 1;
eta = 0.5;      % learning rate
num_epoch = 100000;
loss = zeros(num_epoch,1);

input_data = [0.1 0.1; 0.1 0.9; 0.9 0.1; 0.9 0.9];
output_label = [0.1; 0.9; 0.9; 0.1];

% Weight initialization (Refernce: Xavier and He et al)
w_ij    = randn(num_input, num_hidden)/sqrt(num_input/2);   % 2 x num_hidden
b_j     = randn(num_hidden, 1);   % num_hidden x 1
w_hj    = randn(num_hidden, num_hidden)/sqrt(num_hidden/2);   % num_hidden x num_hidden
w_jk    = randn(num_hidden, num_output)/sqrt(num_hidden/2);   % num_hidden x num_output
b_k     = randn(num_output, 1);   % 1 x 1

s_j     = zeros(num_hidden, 1);
h_prev  = zeros(num_hidden, 1);
s_k     = zeros(num_output, 1);
y_k     = zeros(num_output, 1);
e_k     = zeros(size(output_label));

% Gradient initialization
d_w_ij  = zeros(num_input, num_hidden);   % 2 x num_hidden
d_b_j   = zeros(num_hidden, 1);   % num_hidden x 1
d_w_hj  = zeros(num_hidden, num_hidden);   % num_hidden x num_hidden
d_w_jk  = zeros(num_hidden, num_output);   % num_hidden x num_output
d_b_k   = zeros(num_output, 1);   % 1 x 1

% Training process
for epoch=1:num_epoch

    for data=1:4%size(input_data,1)
        % Forward propagation
        x_data  = input_data(data,:)'; % 2x1
        s_j     = (x_data'*w_ij)' + b_j + (h_prev'*w_hj)';
        h_j     = sigmoid(s_j, num_hidden);
        s_k     = (h_j'*w_jk)' + b_k;
        y_k     = sigmoid(s_k, num_output);
        e_k(data)= output_label(data) - y_k;
        
        % Backpropagation
        for j=1:num_hidden
            d_w_jk(j)   = -e_k(data)*(1-y_k)*y_k*h_j(j);
        end
        
        d_b_k = -e_k(data)*(1-y_k)*y_k;
        
        del_jk_sum1  = 0;
        for h=1:num_hidden
            del_jk_sum1 = del_jk_sum1 + (w_jk(h)*(1-h_j(h))*h_j(h)*h_prev(h));
        end        
        for j=1:num_hidden
            d_w_hj(j)   = -e_k(data)*(1-y_k)*y_k*del_jk_sum1;
        end
        
        del_jk_sum2  = 0;
        for h=1:num_hidden
            del_jk_sum2 = del_jk_sum2 + (w_jk(h)*(1-h_j(h))*h_j(h)*sum(x_data));
        end        
        for i=1:num_input
            d_w_ij(j)   = -e_k(data)*(1-y_k)*y_k*del_jk_sum2;
        end
        
        del_jk_sum3  = 0;
        for h=1:num_hidden
            del_jk_sum3 = del_jk_sum3 + (w_jk(h)*(1-h_j(h))*h_j(h));
        end
        for j=1:num_hidden            
            d_b_j(j)   = -e_k(data)*(1-y_k)*y_k*del_jk_sum3;
        end
        
        % weight update using gradient descent algorithm
        w_ij = w_ij - eta*d_w_ij;
        b_j  = b_j - eta*d_b_j;
        w_hj = w_hj - eta*d_w_hj;
        w_jk = w_jk - eta*d_w_jk;
        b_k  = b_k - eta*d_b_k;
    end
    
    % loss (cost) evaluation
    J = sum(e_k'*e_k)/2;
    loss(epoch) = J;
    
    if rem(epoch, 100) == 0
        fprintf('Step = %d, Cost = %.5f\n', epoch, J);
    end
    
end


% Test process
for data=1:size(input_data,1)
    % Forward propagation
    x_data  = input_data(data,:)'; % 2x1
    s_j     = (x_data'*w_ij)' + b_j + (h_prev'*w_hj)';
    h_j     = sigmoid(s_j, num_hidden);
    s_k     = (h_j'*w_jk)' + b_k;
    y_k     = sigmoid(s_k, num_output);
    output = y_k;
    
    fprintf('Input: %.1f, %.1f, Output: %.2f\n',x_data(1), x_data(2), output);
end

% Plotting the loss
plot(loss, 'LineWidth', 3)
grid on
title('XOR problem result using RNN')
xlabel('Epoch')
ylabel('Loss')