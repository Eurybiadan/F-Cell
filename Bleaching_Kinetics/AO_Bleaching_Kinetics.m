% Robert F Cooper 2018-11-09
%
% This script is designed to calculate the bleaching kinetics of a variable
% number of stimuli. Set up to check bleach percentages for intrinsic
% experiments.

clear;
% close all force;
% tbDeployToolboxes('registered',{'isetbio', 'BrainardLabToolbox','Psychtoolbox-3','SilentSubstitutionToolbox'})

units = 'trolands';
stim_lambda = 550; % in nm
stim_irradiance =  8; % in uW

num_acquisitions = 8;
single_trial_train = [0 1  0;
                      4 5  18];
num_acquisitions = 1;
single_trial_train = [1  0  0;
                      600 601 605];


trial_train = zeros(2, size(single_trial_train,2).*num_acquisitions);
for n=1:num_acquisitions
    
    startind = (n-1).*size(single_trial_train,2)+1;
    endind = startind+(size(single_trial_train,2)-1);
    if n==1
        trial_train(:,startind:endind) = [single_trial_train(1,:);
                                          single_trial_train(2,:)];
    else
        trial_train(:,startind:endind) = [single_trial_train(1,:);
                                          single_trial_train(2,:)+trial_train(2,startind-1)];
    end
end



[~,I] = AOLightLevelConversions_Func(4, stim_lambda, stim_irradiance, true); % Stimulus Intensity in Td

switch (units)
    case 'trolands'
        I_0 = 10^4.3; % in Td %Stimulus intensity that bleaches at the rate of 1/N
                      % 73.7 Td for rods, 20000 Td for cones
        N = 120; % Scaling factor, where 400=rhodopsin, 120=L/M cones.
    case 'isomerizations'
        I_0 = 10^6.4;
        N = 120; % Scaling factor
    otherwise
        error('Unkown input units specified');
end

p = 1; % Percentage of unbleached pigment that we start with. Assume dark adapted.

dt = 0.0001;

dp_dt_deplete = @(p, I) (( (1-p)./N ) - ( (I.*p)./ (N.*I_0) ));
dp_dt_recover = @(p, I)  ( (1-p)./N );


time=0:dt:trial_train(2,end);
bleach_curve = zeros(size(time));
train_ind = 1;

for t=1:length(time)
    
    if time(t)>trial_train(2, train_ind) 
        train_ind = train_ind+1;
    end
    
    if trial_train(1, train_ind) == 1
        bleach_curve(t) = p + dp_dt_deplete(p,I)*dt;
    else
        bleach_curve(t) = p + dp_dt_recover(p,I)*dt;        
    end
    p = bleach_curve(t); % Update our bleach percentage.
end

[highpks,locs]=findpeaks(bleach_curve(time>(max(time)/2)));
[lowpks,locs]=findpeaks(1-bleach_curve(time>(max(time)/2)));

mean_percent_bleach = mean(lowpks);
mean_differential = mean(highpks)-mean(1-lowpks);


figure; 
plot(time, bleach_curve*100); hold on;
plot([time(1) time(end)],[mean(highpks) mean(highpks)]*100, 'k');
plot([time(1) time(end)],[1-mean(lowpks) 1-mean(lowpks)]*100, 'k');
xlabel('Time (s)'); ylabel('Percent Pigment Remaining');
title(['Mean bleach differential: ' num2str(round(mean_differential*100,3)) '% over time with a ' num2str(log10(I)) 'log Td stimulus.']);
axis([0 time(end) -5 105])
