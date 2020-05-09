function qd = avoidanceIK(robot, config, obstaclePosition, dxe)
    % Find x0 and J0
    points      = getPoints(robot, config);     % get all points
    jacobians   = getJacobians(robot, config);  % get all Jacobians

    x0index = getClosestPoint(points, obstaclePosition); % find index of closest point
    x0 = points(:,x0index);                      % Get closest point
    J0 = jacobians(:,:,x0index);                 % Get corresponding Jacobian
    
    % Compute dx0 using potential field
    gammaL = 0;
    dx0 = -gammaL * nablaU(x0, obstaclePosition);
    dx0 = [dx0' 0 0 0]';

    % Find Je
    Je  = jacobians(:,:,7);
    
   
    % Compute qd
    npMove = ( eye(7) - pinv(Je, 1e-10)*Je ) * pinv( J0 * ( eye(7) - pinv(Je)*Je ), 1e-10 ) * ( dx0 - J0*pinv(Je,1e-10)*dxe );
    %% Pseudo inverse based
     %qd = pinv(Je)*dxe + ( eye(7) - pinv(Je, 1e-10)*Je ) * pinv( J0 * ( eye(7) - pinv(Je)*Je ), 1e-10 ) * ( dx0 - J0*pinv(Je,1e-10)*dxe );
    %% Damped least squares
    lambda = 0.05; % lambda = 0.001
    if norm(npMove) > 1e3
        npMove = 0
    end
    %qd = Je'*inv(Je*Je' + lambda^2*eye(6))*dxe;% + npMove;
    
    qd = Je' * linsolve(Je*Je' + lambda^2*eye(6), dxe) + npMove;
    
end