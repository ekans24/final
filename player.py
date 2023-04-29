import numpy as np
import torch
import torchvision
import time
from PIL import Image
from os import path
from torch.serialization import load

# Initalize
NO_PLAYERS = 2
goal_range = np.float32([[0, 75], [0, -75]])
cone_turn_val = 100
best_kart = 'wilber'
model_type = 'image'

# CPU OR GPU?
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# VERY USEFUL
def norm(x):
    return np.linalg.norm(x)

class Team:
    agent_type = model_type

    def __init__(self):
        self.kart_rider = best_kart
        self.initialize_vars()
        self.model = self._load_model()
        self.transform = self._create_transform()

    def new_match(self, team: int, no_players: int) -> list:
        self.team, self.no_players = team, no_players
        self.initialize_vars()
        print("Match Started")
        return [self.kart_rider] * no_players

    def initialize_vars(self):
    
        # Variables related to players
        ZERO = 0
        self.clock = 0
        self.i = 0
        self.cooldown_period = 0
        self.player_seen = 0
        self.player_puck = 0
        self.recovery = 0
        self.play_puck = True


    def _load_model(self):
        """Load the trained model from the specified path."""
        model_path = path.join(path.dirname(path.abspath(__file__)), 'image_agent.pt')
        model = torch.load(model_path).to(device)
        model.eval()
        return model

    def _create_transform(self):
        """Create a transform object for image preprocessing."""
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])

    def detect_puck(self, input_image):
        # Transform the input image
        transformed_image = self.transform(Image.fromarray(input_image)).to(device)
        
        # Detect puck in the image
        predictions = self.model.detect(transformed_image, max_pool_ks=7, min_score=0.2, max_det=15)
        
        # Check if puck is found
        puck_detected = len(predictions) > 0
        
        return puck_detected, predictions


    def update_puck_location(self, is_puck_found, predictions):
        if is_puck_found:
            # Calculate puck location from predictions
            puck_location = np.mean([prediction[1] for prediction in predictions])
            puck_location = puck_location / 64 - 1
            
            # Update puck location and related variables
            if self.play_puck and np.abs(puck_location - self.player_puck) > 0.7:
                puck_location = self.player_puck
                self.play_puck = False
            else:
                self.play_puck = True
            
            self.player_puck = puck_location
            self.player_seen = self.i
        elif self.i - self.player_seen < 4:
            # Keep the previous puck location if it was seen recently
            self.play_puck = False
            puck_location = self.player_puck
        else:
            # Set puck location to None and initiate recovery steps
            puck_location = None
            self.recovery = 10
    
        return puck_location


    def compute_angles_and_distances(self, player_info, location, direction):
        # Calculate front direction and goal-related variables
        kart_val = 'kart'
        front_val = 'front'
        front = np.float32(player_info[kart_val][front_val])[[0, 2]]
        own_goal_direction = goal_range[self.team - 1] - location
        own_goal_distance = norm(own_goal_direction)
        own_goal_direction = own_goal_direction / norm(own_goal_direction)
        
        # Calculate angle to own goal
        own_goal_angle = np.arccos(np.clip(np.dot(direction, own_goal_direction), -1, 1))
        signed_own_goal_degrees = np.degrees(-np.sign(np.cross(direction, own_goal_direction)) * own_goal_angle)
        
        # Calculate variables related to the opponent's goal
        opponent_goal_direction = goal_range[self.team] - location
        opponent_goal_distance = norm(opponent_goal_direction)
        opponent_goal_direction = opponent_goal_direction / np.linalg.norm(opponent_goal_direction)
        
        # Calculate angle to opponent's goal
        opponent_goal_angle = np.arccos(np.clip(np.dot(direction, opponent_goal_direction), -1, 1))
        signed_opponent_goal_degrees = np.degrees(-np.sign(np.cross(direction, opponent_goal_direction)) * opponent_goal_angle)
        
        # Normalize opponent goal distance
        normalized_opponent_goal_distance = ((np.clip(opponent_goal_distance, 10, 100) - 10) / 90) + 1
        
        return own_goal_distance, signed_own_goal_degrees, normalized_opponent_goal_distance, signed_opponent_goal_degrees


    def compute_actions(self, puck_loc, signed_goal_angle, dist_own_goal, signed_own_goal_deg, player_info, goal_dist, puck_found):
        if puck_loc is None:
            # Handle the case when puck_loc is None, for example by assigning a default value
            puck_loc = 0
    
        # Case 1: No cooldown, puck found, and not in recovery mode
        if (self.cooldown_period == 0 or puck_found) and self.recovery == 0:
            angle_min = 20
            angle_max = 120
            
            # Adjust aim point based on signed goal angle and distance to goal
            if angle_min < np.abs(signed_goal_angle) < angle_max:
                dist_weight = 1 / goal_dist ** 3
                aim_point = puck_loc + np.sign(puck_loc - signed_goal_angle / cone_turn_val) * 0.3 * dist_weight
            else:
                aim_point = puck_loc
            
            # Control acceleration and braking based on the last_seen step
            if self.player_seen == self.i:
                kart_val = 'kart'
                type_val = 'velocity'
                brake = False
                speed = 15
                accelerate = 0.75 if norm(player_info[kart_val][type_val]) < speed else 0
            else:
                brake = False
                accelerate = 0
    
        # Case 2: Cooldown active
        elif self.cooldown_period > 0:
            self.cooldown_period -= 1
            brake = False
            accelerate = 0.5
            aim_point = signed_goal_angle / cone_turn_val
    
        # Case 3: Recovery mode
        else:
            if dist_own_goal > 10:
                aim_point = signed_own_goal_deg / cone_turn_val
                accelerate = 0
                brake = True
                self.recovery -= 1
            else:
                self.cooldown_period = 10
                aim_point = signed_goal_angle / cone_turn_val
                accelerate = 0.5
                brake = False
                self.recovery = 0
    
        return aim_point, brake, accelerate

    def act(self, player_state, player_image):
        result = []
    
        for i in range(NO_PLAYERS):
            player_info = player_state[i]
            image = player_image[i]
    
            # Detect puck and update its location
            puck_found, pred = self.detect_puck(image)
            puck_loc = self.update_puck_location(puck_found, pred)
    
            # Reset variables if the kart's velocity is too low for a certain period
            if norm(player_info['kart']['velocity']) < 1:
                if self.clock == 0:
                    self.clock = self.i
                elif self.i - self.clock > 20:
                    self.initialize_vars()
            else:
                self.clock = 0
    
            # Calculate necessary information about the kart and the goal
            front = np.float32(player_info['kart']['front'])[[0, 2]]
            loc = np.float32(player_info['kart']['location'])[[0, 2]]
            direction_act = front - loc
            direction_act = direction_act / norm(direction_act)
    
            # Compute angles and distances
            dist_own_goal, signed_own_goal_deg, goal_dist, signed_goal_angle = self.compute_angles_and_distances(
                player_info, loc, direction_act)
    
            # Compute actions based on puck location, angles, and distances
            aim_point, brake, accelerate = self.compute_actions(
                puck_loc, signed_goal_angle, dist_own_goal, signed_own_goal_deg, player_info, goal_dist, puck_found)
    
            # Set the steering value
            yield_val = 15
            steer = np.clip(aim_point * yield_val, -1, 1)
            
            # Determine if the kart should drift
            highest_drift_val = 0.7
            drift = np.abs(aim_point) > highest_drift_val
            highest_accelerate = 25
    
            # Prepare the player's action data
            player = {
                'steer': signed_goal_angle if self.i < highest_accelerate else steer,
                'acceleration': 1 if self.i < highest_accelerate else accelerate,
                'brake': brake,
                'drift': drift,
                'nitro': False,
                'rescue': False
            }
    
            result.append(player)
    
        self.i += 1
        return result