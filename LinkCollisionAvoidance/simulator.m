%% Get Points, Closest Point, Jacobians
% getPoints(robot, config)
% getClosestPoint(points, obstacle)
% jacobians = getJacobians(robot, config)
clc;clear;
%% Setup: Remove redundant bodies
robot = loadrobot("frankaEmikaPanda");
removeBody(robot,'panda_rightfinger');
removeBody(robot,'panda_leftfinger');
removeBody(robot,'panda_hand');
removeBody(robot,'panda_link8');

%% Setup: Insert obstacle
% Add obstacle
obstaclePosition = [.575 .30 .45]';
obstacle = collisionSphere(0.1);
obstacle.Pose = trvec2tform(obstaclePosition');

%% Get configuration
config = homeConfiguration(robot);
config(6).JointPosition = pi/2;
config = randomConfiguration(robot);

%% Find x0 and J0
points      = getPoints(robot, config);     % get all points
jacobians   = getJacobians(robot, config);  % get all Jacobians

x0index = getClosestPoint(points, obstaclePosition); % find index of closest point
x0 = points(:,x0index);                      % Get closest point
J0 = jacobians(:,:,x0index);                 % Get corresponding Jacobian

%% Compute dx0
gammaL = 0;
dx0 = -gammaL * nablaU(x0, obstaclePosition);
dx0 = [dx0' 0 0 0]';

%% Find xe and Je
Je  = jacobians(:,:,7);
xe  = [0.5545 0 0.7315]';     % LOAD DATA
dxe = [0.5 0.5 0.5 0 0 0]';   % LOAD DATA
%% Compute qd
%thresh = 1e-4;
%Je(abs(Je) < thresh) = 0;   % Round to avoid 'numerical explosion'
%J0(abs(J0) < thresh) = 0;   % Round to avoid 'numerical explosion'
%pinvJe = pinv(Je);
%pinvJe(abs(pinvJe) < thresh) = 0; 

%eyepinvJeJe = eye(7) - pinv(Je)*Je;
%eyepinvJeJe(abs(eyepinvJeJe) < thresh) = 0;
%step_1 = J0 * eyepinvJeJe;
%step_1(abs(step_1) < thresh) = 0;

qd = pinv(Je)*dxe + pinv( J0 * ( eye(7) - pinv(Je)*Je ) ) * ( dx0 - J0*pinv(Je)*dxe )

%% Visualize
show(obstacle);
hold on
show(robot,config);
scatter3(points(1,:),points(2,:),points(3,:),500,...
    'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0 .75 .75]);
hold off
axis equal
lightangle(-45,0);
view(50,25);

%% Visualize Closest Point
v=[obstaclePosition'; x0'];

show(obstacle);
hold on
show(robot,config);
plot3(v(:,1),v(:,2),v(:,3),'r', 'Linewidth',5)
%surf(xs,ys,zs);
%scatter3(x0(1),x0(2),x0(3),2500,...
%    'MarkerEdgeColor','k',...
%        'MarkerFaceColor',[0 .75 .75]);
hold off
axis equal
lightangle(0,0);
view(50,25);