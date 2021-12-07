function [ModMem, init, transition] = get_states(nC, nStates, nE)

init = randi(100,1,nStates);
init = init/sum(init);

t = 0.005;          % avg off diagonal transition matrix entry
transition = randi(100,nStates,nStates);
transition = t*nStates*transition./sum(transition,2)+eye(nStates)*(1 - t*nStates);

ModMem = randi(nE,nC,nStates).*sign(randn(nC,nStates));
end