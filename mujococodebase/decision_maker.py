from dataclasses import Field
import logging
from typing import Mapping

import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.field import FIFAField, HLAdultField
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum
from mujococodebase.navigation.potential_field import PotentialFieldPlanner
from mujococodebase.navigation.team_manager import TeamManager


logger = logging.getLogger()


class DecisionMaker:
    """
    Responsible for deciding what the agent should do at each moment.

    This class is called every simulation step to update the agent's behavior
    based on the current state of the world and game conditions.
    """

    BEAM_POSES: Mapping[type[Field], Mapping[int, tuple[float, float, float]]] ={
        FIFAField: {
            1: (2.1, 0, 0),
            2: (22.0, 12.0, 0),
            3: (22.0, 4.0, 0),
            4: (22.0, -4.0, 0),
            5: (22.0, -12.0, 0),
            6: (15.0, 0.0, 0),
            7: (4.0, 16.0, 0),
            8: (11.0, 6.0, 0),
            9: (11.0, -6.0, 0),
            10: (4.0, -16.0, 0),
            11: (7.0, 0.0, 0),
        },
        HLAdultField: {
            1: (7.0, 0.0, 0),
            2: (2.0, -1.5, 0),
            3: (2.0, 1.5, 0),
        }
    } 

    def __init__(self, agent):
        """
        Creates a new DecisionMaker linked to the given agent.

        Args:
            agent: The main agent that owns this DecisionMaker.
        """
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.is_getting_up: bool = False
        self.ball_lost_timer: int = 0
        
        # Path Planner using Potential Fields
        self.planner = PotentialFieldPlanner(
            k_attractive=1.0, 
            k_repulsive=2.0, 
            rho_zero=2.0
        )
        self.team_manager = TeamManager(agent)

    def update_current_behavior(self) -> None:
        """
        Chooses what the agent should do in the current step.

        This function checks the game state and decides which behavior
        or skill should be executed next.
        """
        if self.agent.world.is_ball_pos_updated:
            self.ball_lost_timer = 0
        else:
            self.ball_lost_timer += 1

        if self.agent.world.playmode is PlayModeEnum.GAME_OVER:
            return

        if self.agent.world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            self.agent.server.commit_beam(
                pos2d=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][:2],
                rotation=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][2],
            )

        if self.is_getting_up or self.agent.skills_manager.is_ready(skill_name="GetUp"):
            self.is_getting_up = not self.agent.skills_manager.execute(skill_name="GetUp")

        elif self.agent.world.playmode is PlayModeEnum.PLAY_ON:
            if self.team_manager.should_go_to_ball():
                self.carry_ball()
            else:
                self.move_to_strategic_position()
        elif self.agent.world.playmode in (PlayModeEnum.BEFORE_KICK_OFF, PlayModeEnum.THEIR_GOAL, PlayModeEnum.OUR_GOAL):
            self.agent.skills_manager.execute("Neutral")
        else:
            if self.team_manager.should_go_to_ball():
                self.carry_ball()
            else:
                self.move_to_strategic_position()

        self.agent.robot.commit_motor_targets_pd()

    def get_obstacles(self):
        """Returns a list of 2D positions of all other robots."""
        obstacles = []
        # Opponents
        for p in self.agent.world.their_team_players:
            if p.last_seen_time and (self.agent.world.server_time - p.last_seen_time < 2.0):
                obstacles.append(p.position[:2])
        # Teammates
        for i, p in enumerate(self.agent.world.our_team_players):
            if (i + 1) != self.agent.world.number: # Don't avoid yourself
                if p.last_seen_time and (self.agent.world.server_time - p.last_seen_time < 2.0):
                    obstacles.append(p.position[:2])
        return obstacles

    def move_to_strategic_position(self):
        """Moves the robot to its assigned strategic position."""
        target_pos = self.team_manager.get_strategic_position()
        my_pos = self.agent.world.global_position[:2]
        ball_pos = self.agent.world.ball_pos_filtered[:2]
        
        obstacles = self.get_obstacles()
        
        next_target = self.planner.get_next_step(
            current_pos=my_pos,
            goal_pos=target_pos,
            obstacles=obstacles,
            step_size=0.5
        )
        
        # Always face the ball when positioning
        desired_orientation = MathOps.vector_angle(ball_pos - my_pos)
        
        self.agent.skills_manager.execute(
            "Walk",
            target_2d=next_target,
            is_target_absolute=True,
            orientation=desired_orientation
        )

    def carry_ball(self):
        """
        Moves the robot toward the goal while handling the ball.
        """
        dist_to_ball = np.linalg.norm(self.agent.world.ball_pos_filtered[:2] - self.agent.world.global_position[:2])
        
        # Only search if ball is lost for more than 10 frames (~0.2s)
        # AND we are not very close to it (if we are close, we assume it's under our feet)
        if self.ball_lost_timer > 10 and dist_to_ball > 0.4:
            # If the ball is not visible, spin in place to find it
            current_yaw = self.agent.robot.global_orientation_euler[2]
            search_orientation = MathOps.normalize_deg(current_yaw + 30)
            
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=self.agent.world.global_position[:2],
                is_target_absolute=True,
                orientation=search_orientation
            )
            return

        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos_filtered[:2]
        my_pos = self.agent.world.global_position[:2]

        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            return 
        ball_to_goal_dir = ball_to_goal / bg_norm

        # Fine-tuned parameters
        dist_from_ball_to_start_carrying = 0.2
        carry_ball_pos = ball_pos - ball_to_goal_dir * dist_from_ball_to_start_carrying

        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        if my_to_ball_norm == 0:
            my_to_ball_dir = np.zeros(2)
        else:
            my_to_ball_dir = my_to_ball / my_to_ball_norm

        cosang = np.dot(my_to_ball_dir, ball_to_goal_dir)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_diff = np.arccos(cosang)

        # REDUCED TOLERANCE for better alignment
        ANGLE_TOL = np.deg2rad(15.0)
        # If very close to the ball, we consider it "controlled"
        ball_at_feet = dist_to_ball < 0.35 
        
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)
        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        # Get obstacles for potential field planning
        obstacles = self.get_obstacles()

        if ball_at_feet and aligned:
            # BALL CONTROL: Move directly to goal
            # We bypass the "preparation point" and go straight to the goal
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=their_goal_pos,
                is_target_absolute=True,
                orientation=desired_orientation
            )
        elif not aligned or not behind_ball:
            # APPROACH: Navigate to the preparation point (behind the ball)
            next_target = self.planner.get_next_step(
                current_pos=my_pos,
                goal_pos=carry_ball_pos,
                obstacles=obstacles,
                step_size=0.5
            )
            
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=next_target,
                is_target_absolute=True,
                orientation=desired_orientation
            )
        else:
            # PUSH: Target the goal directly
            next_target = self.planner.get_next_step(
                current_pos=my_pos,
                goal_pos=their_goal_pos,
                obstacles=obstacles,
                step_size=0.5
            )
            
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=next_target,
                is_target_absolute=True,
                orientation=desired_orientation
            )
