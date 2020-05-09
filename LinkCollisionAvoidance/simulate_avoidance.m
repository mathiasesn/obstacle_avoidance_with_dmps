%% Get Points, Closest Point, Jacobians
% getPoints(robot, config)
% getClosestPoint(points, obstacle)
% jacobians = getJacobians(robot, config)
clc;clear;format compact;

demo = importdata('demo.dat');
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

xe = startPose(1:3,4)';
robotgif = figure
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
    
    if mod(i, 10) == 0
        show(robot, config);
        hold on
        show(obstacle)
        hold off
        drawnow
        view(84,22)
   %    waitfor(r);
        % For generation of GIF
        frame = getframe(1);
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