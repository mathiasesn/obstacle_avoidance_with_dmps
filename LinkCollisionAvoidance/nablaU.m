function value = nablaU(x, obstaclePosition)
    eta = 1;
    p0 = 0.25;
    derivP = (x-obstaclePosition) ./ p(x, obstaclePosition);
    value = eta * (1/p(x,obstaclePosition) - 1/p0)*(-1/(p(x,obstaclePosition)^2))*derivP;
end