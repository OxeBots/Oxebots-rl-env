from dataclasses import Field
import logging
from typing import Mapping
import time
from math import hypot
import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world import world
from mujococodebase.world.field import FIFAField, HLAdultField
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum
from mujococodebase.navigation.potential_field import PotentialFieldPlanner


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

      
    def process_team_messages(self) -> None:
        """
        Process messages received from teammates.

        This function can be expanded to handle different types of messages
        and update the agent's state or behavior accordingly.
        """
        for msg in self.agent.world.team_messages:
            
                # Example: Process ball position messages from teammates
            try:
                if msg.startswith("B:"):
                    msg_content = msg.replace("B:", "").strip()
                    parts = msg_content.split()
                    if len(parts) < 4:
                        logger.warning(f"Mensagem mal formatada: {msg}")
                        continue
                    x = float(parts[0].rstrip(','))
                    y = float(parts[1])
                    timestamp = float(parts[2])
                    sender_id = int(float(parts[3]))

                    self.process_ball_teammate_ball_info(
                        x = x,
                        y = y,
                        timestamp = timestamp, 
                        sender_id = sender_id
                    )
            except Exception as e:
                logger.error(f"Erro ao processar mensagem de time: {msg} - {e}")
                    
        self.agent.world.team_messages.clear()

    def process_ball_teammate_ball_info(self, x: float, y: float, timestamp: float, sender_id: int) -> None:
        logger.info(f"Bola de robô {sender_id}: ({x:.2f}, {y:.2f})")
        #O que o robô vai fazer com essa informação?
        ball_pos = np.array([x, y])
        robot_pos = self.agent.world.global_position[:2]
        dist_to_teammate_ball = np.linalg.norm(ball_pos - robot_pos)

        if dist_to_teammate_ball < 1.0 or not self.agent.world.is_ball_pos_updated:
            self.agent.world.ball_pos_filtered[:2] = ball_pos
            self.agent.world.is_ball_pos_updated = True
        now = time.time()
        if now - timestamp < 0.5:
            self.agent.world.ball_pos_teammate[:2] = ball_pos
            self.agent.world.ball_teammate_timestamp = timestamp
        
        
    def update_ball_info(self) -> None:
        """
        Updates the ball information in the agent's world state.

        This function checks if the ball position has been updated and
        updates the agent's internal state accordingly.
        """
        now = time.time()   
        world = self.agent.world
       #aq eu criei coloquei o mundo na variavel

        if world.is_ball_pos_updated:
            x,y = world.ball_pos[:2]
            send = False;
            
            min_dist = 0.2
            min_time = 0.5
            last_pos = getattr(self, "_last_ball_sent_pos", None)
            last_time = getattr(self, "_last_ball_sent_time", None)

            if last_pos is None:
                send = True
            else:
                dist = hypot(x - last_pos[0], y - last_pos[1])
                if dist >= min_dist:
                    send = True
                elif last_time is not None and (now - last_time) >= min_time:
                    send = True
            if send:
                mensagem  = f"B:{x:.3f}, {y:.3f} {now:.3f} {self.agent.world.number:.3f}"
                self.agent.server.send_team_message(mensagem)
                self._last_ball_sent_time = now
                self._last_ball_sent_pos = (x, y)

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
            self.carry_ball()
        elif self.agent.world.playmode in (PlayModeEnum.BEFORE_KICK_OFF, PlayModeEnum.THEIR_GOAL, PlayModeEnum.OUR_GOAL):
            self.agent.skills_manager.execute("Neutral")
        else:
            self.carry_ball()

        self.agent.robot.commit_motor_targets_pd()
        self.update_ball_info()
        self.process_team_messages()


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

    def carry_ball(self):
        """
        Basic example of a behavior: moves the robot toward the goal while handling the ball.
        """
        dist_to_ball = np.linalg.norm(self.agent.world.ball_pos_filtered[:2] - self.agent.world.global_position[:2])
        
        # Only search if ball is lost for more than 10 frames (~0.2s)
        # AND we are not very close to it (if we are close, we assume it's under our chin)
        if self.ball_lost_timer > 10 and dist_to_ball > 0.5:
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
        dist_from_ball_to_start_carrying = 0.25
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

        ANGLE_TOL = np.deg2rad(5.0)
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)

        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        # Get obstacles for potential field planning
        obstacles = self.get_obstacles()

        if not aligned or not behind_ball:
            # Navigate to the preparation point using Potential Fields
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
            # PUSH: Target the goal directly, but still avoid obstacles if necessary
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

