%% Get Points, Closest Point, Jacobians
% getPoints(robot, config)
% getClosestPoint(points, obstacle)
% jacobians = getJacobians(robot, config)
clc;clear;format compact;

demo = importdata('demo.dat');

%% GIF settings
creategif = 1;  % Set this to 0 if GIF should not be created.
if creategif == 1
    robotgif = figure
    filename = 'devrobot.gif';
    first = 1;
end

%% Setup: Remove redundant bodies
robot = loadrobot("frankaEmikaPanda");
removeBody(robot,'panda_rightfinger');
removeBody(robot,'panda_leftfinger');
removeBody(robot,'panda_hand');
removeBody(robot,'panda_link8');

%% Setup: Insert obstacle
% Add obstacle
obstaclePosition = [.25 .25 .45]';%[.575 .30 .45]';
obstacle = collisionSphere(0.1);
obstacle.Pose = trvec2tform(obstaclePosition');

%% Compute: Trajectory
f = waitbar(0,'Initializing...');
% Get start config
ik = inverseKinematics('RigidBodyTree', robot);
startPose = trvec2tform(demo(1,1:3));
weights = ones(1,6);
homeConfig = homeConfiguration(robot);
config = ik('panda_link7', startPose, weights, homeConfig);
q = [2.8973 -0.9636 -2.5422 -1.4848 2.5051 0.8937 -0.5611]';
% From now on solve IK by moving with qd
%framesPerSecond = 15;
%r = rateControl(framesPerSecond);
demo = importdata('demo.dat');
dt = 0.002;
preqd = zeros(7,1);

qdArray = [];
traj = [];

%   show(robot, config);
%   hold on
%   show(obstacle, [0,0,0])
%   hold off
%   view(84,22)
%   zoom(2)
%   xlim([-0.3 0.8])
%  q = [];
%  for i = 1:7
%      q = [q config(i).JointPosition];
%  end
%  q

q = [2.8973   -0.9636   -2.5422   -1.4848    2.5051    0.8937   -0.5611]
for i = 1:7
    config(i).JointPosition = q(i);
end
% initial config far away
% q1 = [-0.5786    1.3830    1.2284   -1.4300    1.8155    1.2657 -0.7811]
% initial config very close to obstacle
% q2 = [1.0888    1.7628   -1.6201   -1.4280   -1.3737    1.5915    0.3325]
% medium
% q3 = [2.8973   -0.9636   -2.5422   -1.4848    2.5051    0.8937   -0.5611]

xe = startPose(1:3,4)';
for i = 1:size(demo,1)-1
    %dxe = (demo(i+1, 1:6) - demo(i, 1:6)) ./ dt; % Directly from DMP
    
    dxe = [(demo(i+1, 1:3) - xe) (demo(i+1, 4:6)-demo(i,4:6))] ./ dt;% Attraction
    
    %dxe = [dxe 0 0 0]';
    qd = avoidanceIK(robot, config, obstaclePosition, dxe');
    qd = ClampAbsMax(qd, 1);
    qdArray = [qdArray qd];
    if norm(qd) > 10
        disp(i)
        disp(norm(qd))
    end
    q = q + qd * dt;
    for j = 1:7  % Update configuration
        config(j).JointPosition = config(j).JointPosition + qd(j) * dt;
    end
    xe = getTransform(robot,config,'panda_link7');
    xe = tform2trvec(xe);
    traj = [traj xe'];
    
    if mod(i, 10) == 0 && creategif == 1
        show(robot, config);
        hold on
        show(obstacle)
        hold off
        drawnow
        view(84,22)
        zoom(2)                 % TEST
        xlim([-0.3 0.8])        % TEST
        zlim([-0.7 0.7])
        % For generation of GIF
        frame = getframe(robotgif);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if first == 1;
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
            first = 0;
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append');
        end
    end
    if mod(i,100) == 0
        waitbar(i/(size(demo,1)-1),f,'Computing...');
        %disp(sprintf('%d/%d',i,size(demo,1)-1))
    end
     
end
close(f)
%% Plot path
tiledlayout(3,1);

% Tile 1
nexttile
plot((1:4222)*dt,traj(1,:), 'LineWidth', 1.5)
hold on
demoplotX = plot((1:4223)*dt,demo(:,1), 'LineWidth', 4);
hold off
grid on
leg = legend('\boldmath$x_e$','\boldmath${x}_{e,demo}$','location','eastoutside', 'Interpreter', 'Latex');
leg.FontSize = 10;
xlabel('Time [s]')
ylabel('X [m]')
xlim([0 4222*dt])
demoplotX.Color(4) = 0.30;
% Tile 2
nexttile
plot((1:4222)*dt,traj(2,:), 'LineWidth', 1.5)
hold on
demoplotY = plot((1:4223)*dt,demo(:,2), 'LineWidth', 4);
hold off
grid on
xlabel('Time [s]')
ylabel('Y [m]')
xlim([0 4222*dt])
demoplotY.Color(4) = 0.30;


% Tile 3
nexttile
plot((1:4222)*dt,traj(3,:), 'LineWidth', 1.5)
hold on
demoplotZ = plot((1:4223)*dt,demo(:,3), 'LineWidth', 4);
hold off
grid on
xlabel('Time [s]')
ylabel('Z [m]')
xlim([0 4222*dt])
demoplotZ.Color(4) = 0.30;


%% Show joint velocities
dt = 1/500;

figure(1)
plot((1:4222)*dt, qdArray(3,:), 'LineWidth', 1)
grid on
xlim([0 4222*dt])
ylabel('$\dot{q}$ [rad/s]','Interpreter','Latex')
xlabel('Time [s]')
set(gcf,'Position',[100 100 500 300])

figure(2)
plot1 = plot((1:4222)*dt, qdArray(3,:))
%plot1.Color(4) = 0.15;
%hold on
%plot(movmean(qdArray(3,:), 5))
%hold off
ylim([-1 1])
xlim([0 4222*dt])
grid on
ylabel('$\dot{q}$ [rad/s]','Interpreter','Latex')
xlabel('Time [s]')
set(gcf,'Position',[100 100 500 300])

%% Static: Visualize
% show(obstacle);
% hold on
% show(robot,config);
% scatter3(points(1,:),points(2,:),points(3,:),500,...
%     'MarkerEdgeColor','k',...
%         'MarkerFaceColor',[0 .75 .75]);
% hold off
% axis equal
% lightangle(-45,0);
% view(50,25);