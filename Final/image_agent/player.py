import torch
from models import Planner, CNNClassifier
from os import path
import numpy as np
import torchvision
from PIL import Image
from tournament import utils 

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        #change device
        if torch.has_cuda:
            self.device = torch.device('cuda')
        elif torch.has_mps:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print('device: ', self.device)



        self.team = None
        self.num_players = None
        self.goals_center = torch.tensor([[0., 75.], [0., -75.]])
        self.step = 0
        print('-------------------Loading classifier model--------------------')
        self.classifier = CNNClassifier()
        self.classifier.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn1.th')))
        self.classifier.eval()
        print('-------------------Loading planner model--------------------')
        self.planner = Planner()
        self.planner.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))
        self.planner.eval()
        print('planner model loaded........................................')

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['kiki'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """       
        # hyperparameters
        TARGET_VELOCITY = 20.
        DRIFT_THRESHOLD = 10.
        BRAKE_THRESHOLD = DRIFT_THRESHOLD/2.
        
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # initialization
        acceleration = 0.
        steer = 0.
        drift = False
        brake = False
        nitro = False
        rescue = False
        fire = False
        actions = []
        #print('initialization finished..........................')
        for i in range(self.num_players):
            img = player_image[i]
            img_classifier = torchvision.transforms.ToTensor()(img)
            img_planner = img_classifier[None,:]

            # print('orgi image type: ', type(player_image[i]))
            # print('orgi image size: ', player_image[i].shape)
            # print('img type: ', type(img))
            # print('img.shape: ', img.shape)

            kart_sees_puck = self.classifier(img_classifier).to(self.device) > 0
            # print('kart_sees_puck: ', kart_sees_puck)
            # print('planner: ', self.planner.forward(img_planner))
            # exit()
            # print('kart_sees_puck: ', kart_sees_puck)
            # exit()        
            if kart_sees_puck == False:  # no puck detected in the image
                # Rotate back
                kart_actions = dict(acceleration=1, steer=1, brake=brake, drift = drift, nitro=nitro, rescue = rescue, fire=fire)    # rescue?
            #  TODO: if one sees the puck, start with that one
            #  TODO: if both see the puck, start with the one that is closer to the puck
            else:  
                pstate = player_state[i]
                # pimage = player_image[i]
                kart_velocity = pstate['kart']['velocity']
                kart_speed = np.linalg.norm(kart_velocity)
                # TODO: play with y component
                # print('kart_velocity:' , kart_velocity)
                # print('kart speed: ', kart_speed)
                # exit() 
                # features of kart in field coordinates
                #kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
                #kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
                #kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
                #kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
                # features of goal
                #opponent_goal_center = self.goals_center[self.team]     # TODO: check goals
                #puck_to_goal_line = (opponent_goal_center-puck_center) / torch.norm(opponent_goal_center-puck_center)
                #puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0]) 
                #kart_to_goal_line_angle_difference = limit_period((kart_angle - puck_to_goal_line_angle)/np.pi)  
                #kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

                # in image coordinates
                #kart_center = torch.tensor([0.,0.])
                # print(pstate['kart']['location'])
                # print(pstate['camera']['projection'])
                # print(type(pstate['camera']['projection']))
                # exit()
                
                kart_center = utils.PyTux._to_image(pstate['kart']['location'], pstate['camera']['projection'], pstate['camera']['view'])
                # print('kart_center: ', kart_center)
                puck_center = (self.planner(img_planner).to(self.device)).detach().cpu().squeeze()
                # print('puck_center: ', puck_center)
                #exit()                
                kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
                # print('kart_to_puck_direction: ', kart_to_puck_direction)
                kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 
                # print('kart_to_puck_angle: ', kart_to_puck_angle)
                # exit()
                
                # defining actions
                if kart_speed  < TARGET_VELOCITY: acceleration = 1.0
                #if puck_center[0] > 0.:
                steer = float(np.cos(kart_to_puck_angle))
                # else:
                    # steer = -np.cos(kart_to_puck_angle)
                if kart_to_puck_angle < DRIFT_THRESHOLD and kart_speed > TARGET_VELOCITY/2.: 
                    drift = True
                    if kart_to_puck_angle < BRAKE_THRESHOLD: brake = True  
                if  puck_center[0] < 0.1 and acceleration==1:
                    nitro = True
                    #fire = True    
                # TODO: add steps to decide action based on the comparison with the previous step
                # TODO: if stuck, back up
                kart_actions = dict(acceleration=acceleration, steer=steer, brake=brake, drift = drift, nitro=nitro, rescue = rescue, fire=fire)       
            actions.append(kart_actions)
            # print(actions)
        return actions
