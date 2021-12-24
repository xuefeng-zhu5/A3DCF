%% Optimisation (lambda3 is not used, set as 0. No temporal consistency)
function [filter_model_f, P] = train_filter(xlf, feature_info, yf, seq, params, filter_model_f)

if seq.frame == 1
    filter_model_f = cell(size(xlf));
end

for k = 1: numel(xlf)
    model_xf = gather(xlf{k});
    % intialize the variables and parameters
    % <
    if (seq.frame == 1)
        filter_model_f{k} = zeros(size(model_xf));
        lambda3 = 0;
        iter_max = 3;
        mu_max = 20;
    else
        lambda3 = params.lambda3(feature_info.feature_is_deep(k)+1);
        iter_max = 2;
        mu_max = 0.1;
    end
    lambda1 = params.lambda1;
    lambda2 = params.lambda2;
    g_f = single(zeros(size(model_xf)));
    h_f = g_f;
    gamma_f = g_f;
    mu  = 1;
    % >
    
    % pre-compute the variables
    % <
    T = feature_info.data_sz(k)^2;
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    Sfilter_pre_f = sum(conj(model_xf) .* filter_model_f{k}, 3);
    Sfx_pre_f = bsxfun(@times, model_xf, Sfilter_pre_f);
    % >
	Pk = ones(size(model_xf));
    iter = 1;
    while (iter <= iter_max)
      
        B = S_xx + (T*lambda2);
        D = S_xx + (T * mu);
        Shx_f = sum(conj(model_xf) .* h_f, 3);
        Sgx_f = sum(conj(model_xf) .* gamma_f, 3);

 
        % subproblem g
        g_f = ((1/(T*(mu + lambda3)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(mu + lambda3)) * gamma_f) +(mu/(mu + lambda3)) * h_f) - ...
            bsxfun(@rdivide,(1/(T*(mu + lambda3)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) - (1/(mu + lambda3))* (bsxfun(@times, model_xf, Sgx_f)) +(mu/(mu + lambda3))* (bsxfun(@times, model_xf, Shx_f))), D);

        % subproblem h
        h = ifft2((mu * g_f + gamma_f) ./ (lambda1 + mu * T), 'symmetric');


        if iter ==1
            if feature_info.feature_is_deep(k) == 1
                j_r = fft2(bsxfun(@times, h, params.priori_mask));
            else
                j_r = fft2(bsxfun(@times, h, params.mask_window{k}));
            end
        else
            j_r = fft2(h .* Pk);
        end
        Sjx_f = sum(conj(model_xf) .* j_r, 3);
        % subproblem P
        j_f = ((1/(T*lambda2))*bsxfun(@times,yf{k},model_xf)) + j_r - ...
            bsxfun(@rdivide, (1/(T*lambda2)*bsxfun(@times, model_xf, (S_xx .* yf{k})) + (1/mu) * bsxfun(@times, model_xf, Sjx_f)), B);
        
        if seq.frame >1
            j = ifft2(0.05*j_f + 0.95 * filter_model_f{k}, 'symmetric');
        else
            j = ifft2(j_f, 'symmetric');
        end

        Pk = ones(size(model_xf));
        for i = 1 : size(model_xf, 3)
            jc = j(:,:,i);
            [~,b] = sort(abs(jc(:)), 'descend');   
            Pki = Pk(:,:,i);
            Pki(b(ceil(params.spatial_selection_rate(feature_info.feature_is_deep(k)+1)*numel(b)):end)) = 0;
            Pk(:,:,i) = Pki;
        end
        

        if feature_info.feature_is_deep(k) == 1
            Pk = bsxfun(@times, Pk, params.priori_mask);
        else 
            Pk = bsxfun(@times, Pk, params.mask_window{k});
        end 
        
        h_f = fft2(h .* Pk);
        
        
        
        % subproblem Eqn.13.c
        gamma_f = gamma_f + (mu * (g_f - h_f));
        % >
        
        % update the penalty mu
        mu = min(10 * mu, mu_max);
        % >
        
        iter = iter+1;

    end 
    P{k} = Pk;      

    % <
    if seq.frame == 1
        filter_model_f{k} = h_f;        
    else
        filter_model_f{k} = params.learning_rate(feature_info.feature_is_deep(k)+1)* h_f ...
            + (1-params.learning_rate(feature_info.feature_is_deep(k)+1))*filter_model_f{k};
    end
    % >
end    
end






