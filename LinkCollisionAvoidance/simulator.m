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

%% Find x0 and J0
points      = getPoints(robot, config);     % get all points
jacobians   = getJacobians(robot, config);  % get all Jacobians

x0index = getClosestPoint(points, obstaclePosition); % find index of closest point
x0 = points(:,x0index);                      % Get closest point
J0 = jacobians(:,:,x0index);                 % Get corresponding Jacobian

%% Compute dx0

%% Visualize
points = getPoints(robot, config);



show(obstacle)
hold on
show(robot,config)
scatter3(points(1,:),points(2,:),points(3,:),500,...
    'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0 .75 .75])
hold off
axis equal
lightangle(-45,0)
view(50,25)



%% Find closest point
obstaclePos = [0.575 0.30 0.45]

minindex = getClosestPoint(points, obstaclePos)

obstacle = collisionSphere(0.1);
obstacle.Pose = trvec2tform([.575 .30 .45]);


show(robot,config)
hold on
show(obstacle)
scatter3(points(minindex,1),points(minindex,2),points(minindex,3),500,...
    'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0 .75 .75])
hold off