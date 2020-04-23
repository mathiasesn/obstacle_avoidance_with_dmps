function jacobians = getJacobians(robot, config)
    names = robot.BodyNames;
    jacobians = zeros(6,7,7);
    for i = 1:size(names,2) % amount of joints
        name = names{i};
        Ji = geometricJacobian(robot, config, name);     % JACOBIAN
        Ji = [Ji(4:6,:);Ji(1:3,:)];
        %Ji = Ji(4:6,:);
        jacobians(:,:,i) = Ji;
    end
end