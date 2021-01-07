function [g_best_val,valuess, g_best]=HGWOP(fhd,LB,UB,D,iter,swarm_size)%
% Hybrid particle swarm and grey wolf optimizer,HGWOP
%_________________________________________________________________________%
%  HGWOP source codes demo 1.0                                            %
%                                                                         %
%  Developed in MATLAB R2014b                                             %
%                                                                         %
%  Author and programmer: Xinming Zhang                                   %
%   Main paper:
%   Zhang, X.M., Lin, Q.Y., & Mao, W.T.et al (2021). Hybrid Particle Swarm
%   and Grey Wolf Optimizer and its application toclustering optimization.
%   Applied Soft Computing, 2020, 
%   DOI: https://doi.org/10.1016/j.asoc.2020.107061
%   https://www.sciencedirect.com/science/article/abs/pii/S1568494620309996?via%3Dihub
%         e-Mail: xinmingzhang@126.com                                    
Alpha_pos=zeros(1,D);
Alpha_score=inf; %change this to -inf for maximization problems
Beta_pos=zeros(1,D);
Beta_score=inf; %change this to -inf for maximization problems
Delta_pos=zeros(1,D);
Delta_score=inf; %change this to -inf for maximization problems
c1=2;
vmax_coef=0.1;
v_max= vmax_coef* (UB-LB);
v_min=-v_max;
aa=repmat(LB,swarm_size,1);
bb=repmat(UB,swarm_size,1);
particles_x=aa+rand(swarm_size,D).*(bb-aa);
particles_v=repmat(v_min,swarm_size,1)+rand(swarm_size,D).*repmat((v_max-v_min),swarm_size,1);
f_val=feval(fhd,particles_x);
for j=1:swarm_size
    if f_val(j)<Alpha_score
        Alpha_score=f_val(j); % Update alpha
        Alpha_pos=particles_x(j,:);
    end
    if f_val(j)>Alpha_score && f_val(j)<Beta_score
        Beta_score=f_val(j); % Update beta
        Beta_pos=particles_x(j,:);
    end
    if f_val(j)>Alpha_score && f_val(j)>Beta_score && f_val(j)<Delta_score
        Delta_score=f_val(j); % Update delta
        Delta_pos=particles_x(j,:);
    end
end
p_best=particles_x;
p_best_val=f_val;
[~,index]= min(f_val(:,1));
g_best=particles_x(index,:);
g_best_val=f_val(index,1);
g_best_val_t=zeros(1,iter);
valuess=zeros(1,iter*swarm_size);
g_best_t=zeros(1,D);
for i=1:iter    
    cr=0.5*(sin(2*pi*0.25*i+pi)*i/iter+1);      
    ax=2-i*((2)/iter);w=ax/2;
    [p_best_val,Ind]=sort(p_best_val);
    particles_x=particles_x(Ind,:);
    particles_v=particles_v(Ind,:);
    p_best=p_best(Ind,:);
    ss=cumsum(p_best,1);
    ind=randperm(swarm_size);
    for j=1:swarm_size        
        if  ((i>5) && (f_val(j)<=g_best_val_t(i-5)))||(j==1)
            prev_pos=particles_x(j,:);
            particles_x(j,:)=SDPGWOOperator(particles_x(j,:),Alpha_pos,Beta_pos,Delta_pos,p_best,j,ind,D,ax,cr);            
            particles_v(j,:)=prev_pos-particles_x(j,:);
            for k=1:D
                if particles_v(j,k)>v_max(k)
                    particles_v(j,k)=v_max(k);
                end
                if particles_v(j,k)<v_min(k)
                    particles_v(j,k)=v_min(k);
                end
            end            
        else 
            temp=MELOperator(particles_x(j,:),ss,j);
            particles_v(j,:)=w*particles_v(j,:)+c1*rand(1,D).*temp;
             for k=1:D
                if particles_v(j,k)>v_max(k)
                    particles_v(j,k)=v_max(k);
                end
                if particles_v(j,k)<v_min(k)
                    particles_v(j,k)=v_min(k);
                end
            end
            particles_x(j,:)=particles_x(j,:)+particles_v(j,:);                   
        end
    end
    particles_x=ControlBound(particles_x,aa,bb);
    f_val=feval(fhd,particles_x);   
    for j=1:swarm_size
        if f_val(j)<p_best_val(j,1)
            p_best(j,:)=particles_x(j,:);
            p_best_val(j,1)=f_val(j);
        end
        if p_best_val(j,1)<g_best_val
            g_best=particles_x(j,:);
            g_best_val=p_best_val(j,1);
        end
        if f_val(j)<Alpha_score
            Alpha_score=f_val(j); % Update alpha
            Alpha_pos=particles_x(j,:);
        end
        if f_val(j)>Alpha_score && f_val(j)<Beta_score
            Beta_score=f_val(j); % Update beta
            Beta_pos=particles_x(j,:);
        end
        if f_val(j)>Alpha_score && f_val(j)>Beta_score && f_val(j)<Delta_score
            Delta_score=f_val(j); % Update delta
            Delta_pos=particles_x(j,:);
        end
    end
    g_best_t(i,:)=g_best;
    g_best_val_t(i)=g_best_val;   
    valuess((i-1)*swarm_size+1:i*swarm_size)=g_best_val;
end
valuess=valuess(1:swarm_size:iter*swarm_size);