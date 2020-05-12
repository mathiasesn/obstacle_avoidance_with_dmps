function qdclamp = ClampAbsMax(qd, d)
    if norm(qd) > d
        qdclamp = qd/norm(qd) * d;
    else
        qdclamp = qd;
    end
end