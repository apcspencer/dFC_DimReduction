% Code adapted from Allen et al., "Tracking whole-brain connectivity dynamics in the resting state." Cereb Cortex. 2014 24(3):663-76. 
% Dependency: SimTB toolbox (https://trendscenter.org/trends/software/simtb/)

%%
clear all; close all; clc
addpath(genpath('simtb_v18')) % path to SimTB toolbox directory

mkdir synthetic; mkdir synthetic/ground_truth; mkdir synthetic/states % output directories
output_folder = 'synthetic/';

%% Edit these settings:
nsubjs = 50;                    % number of subjects
noise_level = 'high';           % noise levels (high/medium/low)
nC = 25;                        % number of nodes
hrf_type = 1;                   % 1 = canonical HRF; 2 = windkessel-balloon
nStates = 5;                    % number of states
TR = 2;                         % repetition time
nT = 540/TR;                    % number of time points/ duration (s)
pState = .5;                    % probability of state specific events 

%%

% Number of event types (i.e., number of different modules)
switch nC
    case 15
        nE = 5;
    case 25
        nE = 7;
    case 50 
        nE = 9;
end

%Module membership for each state
[ModMem, init, transition] = get_states(nC, nStates, nE);


%% Create figure of the connectivity matrix for each state
F = figure('color','w','Name', 'sim_neural_connectivity');

for ii = 1:nStates
    CM = zeros(nC,nC);
    for jj = 1:nC
        for kk = 1:nC
            if ModMem(jj,ii) == ModMem(kk,ii)
                CM(jj,kk) = 1;
            elseif abs(ModMem(jj,ii)) == abs(ModMem(kk,ii))
                CM(jj,kk) = -1;
            else
                CM(jj,kk) = 0;
            end
        end
    end
    
    [Ci,~] = modularity_und(CM);
    [~,Mi] = sort(Ci);
    CM = CM(Mi,Mi);
    ModMem(:,ii) = ModMem(Mi,ii);
    dlmwrite(['synthetic/states/state' num2str(ii) '.csv'],CM*0.8)
    
    subplot(1,nStates,ii)
    H = simtb_pcolor(1:nC, 1:nC, .8*CM);
    axis square; 
    axis ij
    set(gca, 'XTick', [], 'YTick', [], 'CLim', [-1 1])%, 'XColor', [1 1 1], 'YColor', [1 1 1])
    c = get(gca, 'Children');
    set(c(find(strcmp(get(c, 'Type'),'line'))), 'Color', 'w');
    title(sprintf('State %d', ii))
end


%% Create the event time courses

switch noise_level
    case 'low'
        mean_unique_p = 0.3;
        std_unique_p = 0.2;
        mean_unique_a = 0.3;
        std_unique_a = 0.2;
        mean_noise = 0.3;
        std_noise = 0.2;
    case 'medium'
        mean_unique_p = 0.5;
        std_unique_p = 0.3;
        mean_unique_a = 0.5;
        std_unique_a = 0.3;
        mean_noise = 0.4;
        std_noise = 0.2;
    case 'high'
        mean_unique_p = 0.6;
        std_unique_p = 0.2;
        mean_unique_a = 0.7;
        std_unique_a = 0.2;
        mean_noise = 0.6;
        std_noise = 0.2;
end

dwell_time = zeros(nsubjs,nStates);
fractional_occupancy = zeros(nsubjs,nStates);

all_Cdwell = [];

for s = 1:nsubjs
    disp(s)
    
    % sample a different amplitue and probability of unique events for each
    % subject
    pU = mean_unique_p + std_unique_p*randn(1);
    aU = mean_unique_a + std_unique_a*randn(1);
    noise = mean_noise + std_noise*randn(1);

    Sorder = [];
    Cdwell = [0];

    fo = zeros(nStates,1);
    dtc = zeros(nStates,1);
    
    % sample time series
    rng('shuffle')
    state_tseries = zeros(nT,1);
    this_state = randsample_dist(init);
    state_tseries(1) = this_state;
    fo(this_state) = fo(this_state) + 1;
    dtc(this_state) = dtc(this_state) + 1;
    Sorder = cat(2,Sorder,this_state);
    for t = 2:nT
        next_state = randsample_dist(transition(this_state,:));
        state_tseries(t) = next_state;
        if next_state~=this_state
            Sorder = cat(2,Sorder,next_state);
            Cdwell = cat(2,Cdwell,t);
            this_state = next_state;
            dtc(this_state) = dtc(this_state) + 1;
        end
        fo(this_state) = fo(this_state) + 1;
    end
    Cdwell = cat(2,Cdwell,nT);

    dwell_time(s,:) = fo./dtc;
    fractional_occupancy(s,:) = fo/nT;
    dwell_time(isnan(dwell_time)) = 0;

    % random aspects (different for each component)
    eT = rand(nT, nC) < pU;
    eT = eT.*sign(rand(nT, nC)-0.5);
    eT = eT*aU;

    STATE = zeros(1,nT); % state vector
    for ii = 1:length(Sorder)
        sIND = Cdwell(ii)+1:Cdwell(ii+1);
        % events related to each module
        e = rand(length(sIND),nE) < pState;
        e = e.*sign(rand(length(sIND), nE)-0.5);
        for cc = 1:nC
            eT(sIND,cc) = eT(sIND,cc) + sign(ModMem(cc,Sorder(ii)))*e(:,abs(ModMem(cc,Sorder(ii))));
        end
        STATE(sIND) = Sorder(ii);
    end

    % event time series are stored in eT
    %% Convolve event TCs
    TC  = zeros(nT,nC);
    for cc = 1:nC
        TC(:,cc) = simtb_TCsource(eT(:,cc), TR, hrf_type);
    end

    % Add a little gaussian noise
    TC = TC + noise*randn(nT,nC);

    dlmwrite([output_folder num2str(s) '.csv'], TC', 'delimiter',',')
    dlmwrite([output_folder 'ground_truth/' num2str(s) '.csv'], state_tseries, 'delimiter',',')

    all_Cdwell = cat(2,all_Cdwell,TR*(Cdwell(2:end)-Cdwell(1:end-1)));
end

figure;
histogram(all_Cdwell)
disp(['Median ISI = ' num2str(median(all_Cdwell))])

%% Figure to display the states, TCs, and correlation matrices for each partition
F=figure('color','w','Name', 'sim_TCs_CorrMatrices'); 
subplot(4, length(Sorder), 1:length(Sorder))
plot((0:nT-1)*TR, STATE , 'k', 'Linewidth', 1); axis tight; box off
ylabel('State')
set(gca, 'YTick', 1:nStates, 'XTick', Cdwell*TR, 'TickDir', 'out', 'Layer', 'Bottom'); grid on

subplot(4, length(Sorder), length(Sorder)+1:length(Sorder)*2)
plot((0:nT-1)*TR, TC, 'LineWidth',0.75);
xlabel('Time (s)')
ylabel('Amplitude')
set(gca, 'TickDir', 'out', 'XTick', Cdwell*TR, 'Xgrid', 'on'); 
axis tight; box off

for ii = 1:length(Sorder)
    subplot(4,length(Sorder),length(Sorder)*3+ii)
    sIND = Cdwell(ii)+1:Cdwell(ii+1);
    temp = corr(TC(sIND,:));
    H = simtb_pcolor(1:nC, 1:nC, temp);
    axis square; axis ij 
    set(gca, 'XTick', [], 'YTick', [], 'CLim', [-1 1])%, 'XColor', [1 1 1], 'YColor', [1 1 1])
    c = get(gca, 'Children');
    set(c(find(strcmp(get(c, 'Type'),'line'))), 'Color', 'w');        
    text(1.5,-2,sprintf('Partition %d\nState %d', ii, Sorder(ii)), 'Fontsize', 8);
end
