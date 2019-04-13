function [Recog_rate, benar] = ANNBackprop_full(input, target, n_hidden)

[pl,ok] = size(input);[ji,hu] = size(target);
%STEP -1 pembagian data training dan test(50/50)
input_test = input(1:pl/2,:);
target_test = target(1:ji/2,:);
input_train = input((pl/2)+1:pl,:);
target_train = target((ji/2)+1:ji,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TRAIN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%STEP 0 Inisialisasi % alfa
% inisialisasi beban
p = n_hidden;
[n,b] = size(input_train);
    %n = jumlah data input
    %b = banyaknya vektor data input
[m,a] = size(target_train);
    %m = jumlah target
    %a = banyaknya vektor target
beta = 0.7 .* (p)^(1/(n/2));
alpha = 0.4;
kondisi = 0;
inc = 1;
max_epoch = 5000;
epoch = 1;
miu = 0.3;
deltaW_lama = zeros(p,a);
deltaV_lama = zeros(b,p);
deltaW0_lama = zeros(1,a);
deltaV0_lama = zeros(1,p);
E_b = 50;

%Normalisasi input dan target
for mm = 1:n
    for ha = 1:b
        Norm_in(mm,ha) = ( (input_train(mm,ha) - min(input_train(:,ha)))*(1) )/( max(input_train(:,ha)) - min(input_train(:,ha)) );
    end
end
for mm = 1:m
    for ha = 1:a
        Norm_tr(mm,ha) = ( (target_train(mm,ha) - min(target_train(:,ha)))*(1) )/( max(target_train(:,ha)) - min(target_train(:,ha)) );
    end
end
input_train = Norm_in;
target_train = Norm_tr;

%%algoritma nguyen
Vij = (0.5-(-0.5)).*rand(b,p) - 0.5; %random bobot dari -0.5 sampe 0.5
Wjk = (0.5-(-0.5)).*rand(p,a) - 0.5; %random bobot dari -0.5 sampe 0.5

%bobot V
Vj_abs = sqrt(sum(Vij.^2));

for i = 1:b
    for j = 1:p
        V(i,j) = beta.*Vij(i,j)./Vj_abs(j);
    end
end

for j = 1:p
    V0(1,j) = (beta.*2).*rand(1) + -beta;
end

%bobot W
Wk_abs = sqrt(sum(Wjk.^2));

for k = 1:a
    for j = 1:p
        W(j,k) = beta.*Wjk(j,k)./Wk_abs(k);
    end
end

for k = 1:a
    W0(1,k) = (beta.*2).*rand(1) - beta;
end

W_b = W;
V_b = V;
W0_b = W0;
V0_b = V0;

while kondisi == 0 && epoch ~= max_epoch+1

    %step 3
    X = input_train(inc,:);
    T = target_train(inc,:);
    
    %step 4
    for j = 1:p
        for i = 1:b
            sigz(i,j) = X(i)*V(i,j);
        end
    end
    Z_in = V0 + sum(sigz);
    
    for j = 1:p
        Z(1,j) = 1 ./( 1 + exp( -Z_in(1,j)) );
    end
    
    %step 5
    for k = 1:a
        for j = 1:p
            sigy(j,k) = Z(j)*W(j,k);
        end
    end
    Y_in = W0 + sum(sigy);
    
    for k = 1:a
        Y(1,k) = 1./( 1 + exp( -Y_in(1,k)) );
    end
    
    %momentum
    MOMW = miu * deltaW_lama;
    MOMV = miu * deltaV_lama;
    MOMW0 = miu * deltaW0_lama;
    MOMV0 = miu * deltaV0_lama;
    
    %step 6
    for k = 1:a
        dok(1,k) = (T(1,k)-Y(1,k)).* (Y(1,k).*(1 - Y(1,k))) ;
    end                                              %do Y/do Y_in       do Y_in/do W
    
    for j = 1:p
        for k = 1:a
           deltaW(j,k) = alpha.*dok(1,k).*Z(j) + MOMW(j,k);
           deltaW0(1,k) = alpha.*dok(1,k) + MOMW0(1,k);
        end
    end
    
    %step 7
    for j = 1:p
        do_in(1,j) = sum(sum(dok).*W(j,:));
    end
    
    for j = 1:p
        doj(1,j) = do_in(1,j) .* (Z(1,j).*(1 - Z(1,j)));
    end
    
    for i = 1:b
        for j = 1:p
            deltaV(i,j) = alpha.*doj(1,j).*X(i) + MOMV(i,j);
            deltaV0(1,j) = alpha.*doj(1,j) + MOMV0(1,j);
        end
    end
    
    %step 8
    W = W + deltaW;
    W0 = W0 + deltaW0;
    
    V = V + deltaV;
    V0 = V0 + deltaV0;
    
    %step 9 %Toleransi error
    E(1,inc) = 0.5.*sum((T-Y).^2);
    
    if inc == n
        E_tot(1,epoch) = sum(E)/(n);
        if E_tot(1,epoch)>E_b
            W = W_b;
            V = V_b;
            W0 = W0_b;
            V0 = V0_b;
            E_tot(1,epoch) = E_b;
        end
        if E_tot(1, epoch) <= 0.0001
            kondisi = 1;
        else
            kondisi = 0;
        end
        E_b = E_tot(1,epoch);
    end

    if inc == n
        inc = 0;
        epoch = epoch +1;
    end
    
    deltaW_lama = deltaW;
    deltaV_lama = deltaV;
    deltaW0_lama = deltaW0;
    deltaV0_lama = deltaV0;
    W_b = W;
    V_b = V;
    W0_b = W0;
    V0_b = V0;
    inc = inc +1;
end

figure(1);plot(E_tot);xlabel('epoch');ylabel('Error')
disp(E_tot(1, epoch-1))
save('bobot.mat', 'V', 'W')
save('bias.mat', 'V0', 'W0')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kondisi = 0;
inc = 1;
benar = 0;
salah = 0;

%Normalisasi input dan target
for mm = 1:n
    for ha = 1:b
        Norm_in1(mm,ha) = ( (input_test(mm,ha) - min(input_test(:,ha)))*(1-0) )/( max(input_test(:,ha)) - min(input_test(:,ha)) );
    end
end
for mm = 1:m
    for ha = 1:a
        Norm_tr1(mm,ha) = ( (target_test(mm,ha) - min(target_test(:,ha)))*(1-0) )/( max(target_test(:,ha)) - min(target_test(:,ha)) );
    end
end
input_test = Norm_in1;
target_test = Norm_tr1;

while kondisi == 0 

    %step 3
    X = input_test(inc,:);
    T = target_test(inc,:);
    
    %step 4
    for j = 1:p
        for i = 1:b
            sigz(i,j) = X(i)*V(i,j);
        end
    end
    Z_in = V0 + sum(sigz);
    
    for j = 1:p
        Z(1,j) = 1./( 1+exp( -Z_in(1,j)) );
    end
    
    %step 5
    for k = 1:a
        for j = 1:p
            sigy(j,k) = Z(j)*W(j,k);
        end
    end
    Y_in = W0 + sum(sigy);
    
    for k = 1:a
        Y(1,k) = 1./( 1+exp( -Y_in(1,k)) );
    end
    
    %tebak-tebakan
    for k = 1:a
        if Y(1,k) == max(Y)
           Y(1,k) = max(T);
        end
        
        if Y(1,k) ~= max(Y)
           Y(1,k) = min(T);
        end
    end
    
    if Y-T == zeros(1,a)
        benar = benar + 1;
    end
    
    if Y-T ~= zeros(1,a)
        salah = salah + 1;
    end
    
    if inc == n
        kondisi = 1;
    end
    
    inc = inc +1;
end

Recog_rate = benar/n;

end

