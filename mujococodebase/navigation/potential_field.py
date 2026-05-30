import numpy as np

class PotentialFieldPlanner:
    """
    Path planning using Artificial Potential Fields.
    Objects exert attractive (goal) or repulsive (obstacles) forces.
    """
    def __init__(self, k_attractive=1.0, k_repulsive=5.0, rho_zero=1.5):
        self.k_attractive = k_attractive
        self.k_repulsive = k_repulsive
        self.rho_zero = rho_zero # Distance of influence of obstacles

    def get_force(self, current_pos, goal_pos, obstacles):
        """
        Calculates the resultant force vector at the current position.
        
        :param current_pos: np.array([x, y])
        :param goal_pos: np.array([x, y])
        :param obstacles: list of np.array([x, y])
        :return: force_vector np.array([x, y])
        """
        # Attractive force to goal
        f_attractive = self.k_attractive * (goal_pos - current_pos)
        
        # Repulsive force from obstacles
        f_repulsive = np.zeros(2)
        for obs in obstacles:
            dist = np.linalg.norm(current_pos - obs)
            if dist < self.rho_zero and dist > 0.01:
                # Force magnitude increases as distance decreases
                rep_mag = self.k_repulsive * (1.0/dist - 1.0/self.rho_zero) * (1.0/dist**2)
                rep_dir = (current_pos - obs) / dist
                f_repulsive += rep_mag * rep_dir
                
        return f_attractive + f_repulsive

    def get_next_step(self, current_pos, goal_pos, obstacles, step_size=0.5):
        """Calculates the next target point."""
        force = self.get_force(current_pos, goal_pos, obstacles)
        force_norm = np.linalg.norm(force)
        
        if force_norm > 0:
            direction = force / force_norm
            return current_pos + direction * step_size
        return current_pos
