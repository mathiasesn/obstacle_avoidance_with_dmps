function pointIndex = getClosestPoint(points, obstaclePos)
    minvalue = realmax;
    pointIndex = -1;
    for i = 1:size(points, 2) % amount of joints
        currentdist = norm(obstaclePos - points(:,i));
        if currentdist < minvalue
            minvalue = currentdist;
            pointIndex = i;
        end
    end
end