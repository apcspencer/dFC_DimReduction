% Arthur Spencer 24/07/20
% draw a random sample from the discrete distribution with probabilities
% given by the elements of dist. e.g. [0.5 0.2 0.3] gives 50% chance of 1,
% 20% chance of 2 and 30% chance of 3.

function ind = randsample_dist(dist)

if int16(sum(dist))~=1
    error('Distribution does not sum to 1')
end

% make an array in which the number of instances of each sample is defined
% by the distribution
dist = int16([0 cumsum(dist)*1000]);
samples = zeros(1000,1);
for i = 1:length(dist)-1
    samples(dist(i)+1:dist(i+1)) = i*ones(dist(i+1)-dist(i),1);
end

% draw sample
ind = samples(randi(1000,1));