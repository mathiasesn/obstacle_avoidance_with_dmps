clc;clear;format compact;

%% Plot basis functions locations
alpha = 18
n_bfs = 30
c = 1 - exp(-alpha/2 * linspace(0, 1, n_bfs))

h = 1.0 ./ gradient(c)*gradient(c)';
sigma = 0.1; % SIGMA

fplot(@(x) exp(-1/(2*sigma^2)*(x - c(1))^2))
hold on
    for i = 2:n_bfs
        fplot(@(x) exp(-1/(2*sigma^2)*(x - c(i))^2))
    end
hold off
xlim([0 1])
ylim([0 1])
xticks(0:.25:1)
xticklabels(1:-.25:0)
xlabel('\chi')
set(gcf,'Position',[100 100 400 200])