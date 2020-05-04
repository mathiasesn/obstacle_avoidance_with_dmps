function points = getPoints(robot, config)
    names = robot.BodyNames;
    for i = 1:size(names,2) % amount of joints
        name = names{i};
        tf = getTransform(robot, config, names{i});     % JOINT POSITION
        points(1:3,i) = tf(1:3,4);
    end
end