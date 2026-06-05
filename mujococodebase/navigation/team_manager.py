import numpy as np
from enum import Enum, auto

class Role(Enum):
    GOALKEEPER = auto()
    DEFENDER = auto()
    ATTACKER = auto()

class TeamManager:
    def __init__(self, agent):
        self.agent = agent
        self.role = self._assign_role(agent.world.number)
        self.is_currently_going_to_ball = False
        self.hysteresis_advantage = 0.5  # 0.5 meters advantage for the current pursuer

    def _assign_role(self, number: int) -> Role:
        if number == 1:
            return Role.GOALKEEPER
        elif number in [2, 3]:
            return Role.DEFENDER
        else:
            return Role.ATTACKER

    def should_go_to_ball(self) -> bool:
        """
        Decides if this robot should approach the ball based on its role, distance, and hysteresis.
        """
        world = self.agent.world
        my_pos = world.global_position[:2]
        ball_pos = world.ball_pos_filtered[:2]
        
        # Goalkeeper never goes for the ball
        if self.role == Role.GOALKEEPER:
            self.is_currently_going_to_ball = False
            return False

        # Calculate my distance to ball
        my_dist = np.linalg.norm(my_pos - ball_pos)
        
        # Apply hysteresis: if I was already going, I act as if I am closer than I am
        effective_my_dist = my_dist
        if self.is_currently_going_to_ball:
            effective_my_dist -= self.hysteresis_advantage

        # Check teammates' distances
        for i, teammate in enumerate(world.our_team_players):
            teammate_number = i + 1
            if teammate_number == world.number:
                continue
            
            # Skip if we haven't seen the teammate recently
            if not teammate.last_seen_time or (world.server_time - teammate.last_seen_time > 2.0):
                continue

            # Skip the goalkeeper in comparison
            if teammate_number == 1:
                continue

            teammate_pos = teammate.position[:2]
            teammate_dist = np.linalg.norm(teammate_pos - ball_pos)

            # If a teammate is significantly closer (considering my hysteresis), I stay back.
            if teammate_dist < effective_my_dist:
                self.is_currently_going_to_ball = False
                return False
            
            # Tie-breaking for very similar distances
            if abs(teammate_dist - effective_my_dist) < 0.1 and teammate_number < world.number:
                self.is_currently_going_to_ball = False
                return False
        
        # Special role constraints
        if self.role == Role.DEFENDER:
            # Defenders only go if ball is in our half (x < 0)
            if ball_pos[0] > 0:
                self.is_currently_going_to_ball = False
                return False

        self.is_currently_going_to_ball = True
        return True

    def get_strategic_position(self) -> np.ndarray:
        """
        Returns a target position when the robot is NOT going for the ball.
        """
        world = self.agent.world
        ball_pos = world.ball_pos_filtered[:2]
        field_len = world.field.get_length()
        field_wid = world.field.get_width()

        if self.role == Role.GOALKEEPER:
            # Stay in front of the goal, fixed
            goal_pos = world.field.get_our_goal_position()
            return np.array([goal_pos[0] + 1.0, 0.0])

        elif self.role == Role.DEFENDER:
            # Defenders stay at the edge of the penalty area or following the ball's Y
            target_x = -field_len/3  # Fixed defensive line
            target_y = np.clip(ball_pos[1], -field_wid/4, field_wid/4)
            
            if self.agent.world.number == 2:
                target_y += field_wid/8
            else:
                target_y -= field_wid/8
            return np.array([target_x, target_y])

        else: # ATTACKER
            # Spread out in the attack, but stay behind the ball if it's too far
            target_x = max(0.0, ball_pos[0] - 5.0)
            
            # Different positions for attackers
            offsets = {
                4: (0, field_wid/4),
                5: (0, -field_wid/4),
                6: (-2, field_wid/8),
                7: (-2, -field_wid/8)
            }
            off_x, off_y = offsets.get(self.agent.world.number, (0, 0))
            return np.array([target_x + off_x, off_y])
