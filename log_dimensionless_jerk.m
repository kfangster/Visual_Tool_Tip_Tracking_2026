function ldl = log_dimensionless_jerk(speed, fs)
    % Calculates the smoothness metric for the given speed profile using the
    % dimensionless jerk metric (normalized by maximum speed).
    %
    % Parameters:
    %   speed : array
    %       The array containing the movement speed profile.
    %   fs : float
    %       The sampling frequency of the data.
    %
    % Returns:
    %   ldl : float
    %       The log dimensionless jerk estimate of the given movement's smoothness.
    %
    
    % Ensure movement is a column vector
    speed = speed(:);
    
    % Compute scale factor and jerk
    movement_peak = max(abs(speed));
    dt = 1 / fs;
    movement_dur = length(speed) * dt;
    jerk = diff(speed, 2) / dt^2;
    scale = (movement_dur^3) / (movement_peak^2);
    
    % Compute dimensionless jerk
    dl = -scale * sum(jerk.^2) * dt;

    % Compute log of absolute dimensionless jerk
    ldl = -log(abs(dl));
end
