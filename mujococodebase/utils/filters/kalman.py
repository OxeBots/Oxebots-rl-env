import numpy as np

class KalmanFilter:
    """
    A generic implementation of the Kalman Filter.
    Can be used for tracking position and velocity of any object (Ball, Self, Others).
    """
    def __init__(self, dt: float, state_dim: int, obs_dim: int):
        self.dt = dt
        
        # State vector [x, y, vx, vy, ...]
        self.x = np.zeros((state_dim, 1))
        
        # State transition matrix
        self.F = np.eye(state_dim)
        
        # Measurement matrix
        self.H = np.zeros((obs_dim, state_dim))
        
        # Covariance matrix
        self.P = np.eye(state_dim) * 1000.0
        
        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(obs_dim) * 1.0

    def predict(self):
        """Predict the next state."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z: np.ndarray):
        """Update the state with a new measurement z."""
        y = z - np.dot(self.H, self.x) # Innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman Gain
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
