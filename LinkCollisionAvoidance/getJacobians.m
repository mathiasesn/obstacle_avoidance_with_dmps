function jacobians = getJacobians(robot, config)
    names = robot.BodyNames;
    jacobians = zeros(6,7,7);
    for i = 1:size(names,2) % amount of joints
        name = names{i};
        Ji = geometricJacobian(robot, config, name);     % JACOBIAN
        jacobians(:,:,i) = Ji;
    end
end