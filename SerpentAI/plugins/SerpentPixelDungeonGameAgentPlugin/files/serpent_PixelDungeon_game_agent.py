from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

import serpent.cv
import serpent.utilities

from serpent.frame_grabber import FrameGrabber

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace


from .helpers.better_dqn import BDQN


import time
import sys
import collections
import os

import pytesseract


import gc

#import pyperclip

import numpy as np

import skimage.io
import skimage.filters
import skimage.morphology
import skimage.measure
import skimage.draw
import skimage.segmentation
import skimage.color

from datetime import datetime

import offshoot
from serpent.sprite import Sprite
from serpent.sprite_locator import SpriteLocator

from serpent.input_controller import MouseButton

from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

class SerpentPixelDungeonGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.game_state = None
        self._reset_game_state()
        
        
    def setup_play(self):

        plugin_path = offshoot.config["file_paths"]["plugins"]
        '''
        context_classifier_path = f"{plugin_path}/SerpentPixelDungeonGameAgentPlugin/files/ml_models/context_classifier.model"
        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(100,100, 0))  # Replace with the shape (rows, cols, channels) of your captured context frames

        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)
        self.machine_learning_models["context_classifier"] = context_classifier
        '''
        image_data = skimage.io.imread(f"{plugin_path}/SerpentPixelDungeonGamePlugin/files/data/sprites/rats/rat1.png")[...,np.newaxis]
        sprite = Sprite("MY SPRITE",image_data=image_data)
        #extra_image = skimage.io.imread(f"{plugin_path}/SerpentPixelDungeonGamePlugin/files/data/sprites/rats/rat2.png")[...,np.newaxis]
        #sprite.append_image_data(extra_image[..., np.newaxis])
        
        input_mapping = {
        	"UP": [KeyboardKey.KEY_UP],
        	"LEFT": [KeyboardKey.KEY_LEFT],
        	"DOWN": [KeyboardKey.KEY_DOWN],
        	"RIGHT": [KeyboardKey.KEY_RIGHT],
        	#"ATTACK": [KeyboardKey.KEY_A],
        	#"SHOOT": [KeyboardKey.KEY_Q]
        	"UP_BUDDY": [KeyboardKey.KEY_Z],
        	"LEFT_BUDDY": [KeyboardKey.KEY_C],
        	"DOWN_BUDDY": [KeyboardKey.KEY_X],
        	"RIGHT_BUDDY": [KeyboardKey.KEY_V],
        	"INTERACT CELL_BUDDY": [KeyboardKey.KEY_B]            

        }


        
        self.key_mapping = {
        	KeyboardKey.KEY_UP.name : "MOVE UP",
        	KeyboardKey.KEY_LEFT.name : "MOVE LEFT",
        	KeyboardKey.KEY_DOWN.name : "MOVE DOWN",
        	KeyboardKey.KEY_RIGHT.name : "MOVE RIGHT",
        	#KeyboardKey.KEY_A.name : "MELEE ATTACK",
        	#KeyboardKey.KEY_Q.name : "RANGE ATTACK",
        	KeyboardKey.KEY_Z.name : "MOVE UP BUDDY",
        	KeyboardKey.KEY_C.name : "MOVE LEFT BUDDY",
        	KeyboardKey.KEY_X.name : "MOVE DOWN BUDDY",
        	KeyboardKey.KEY_V.name : "MOVE RIGHT BUDDY",
        	KeyboardKey.KEY_B.name : "INTERACT CELL BUDDY"            
        }



        movement_action_space = KeyboardMouseActionSpace(
        	directional_keys=[None, "UP", "LEFT", "DOWN", "RIGHT"]#, "ATTACK", "SHOOT"]
        )


        #Buddy code             
        movement_action_space_buddy = KeyboardMouseActionSpace(
        	directional_keys=[None,"UP_BUDDY", "LEFT_BUDDY", "DOWN_BUDDY", "RIGHT_BUDDY","INTERACT CELL_BUDDY"]
        )



        descendingImage = skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/frame_1541616861.6503916.png")
        self.descIm = serpent.cv.extract_region_from_image(
                 descendingImage,
                 self.game.screen_regions["Descending"]
	    )
        
        mainCharImage = skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/frame_1541616846.6041803.png")
        self.mainChr = serpent.cv.extract_region_from_image(
                 mainCharImage,
                 self.game.screen_regions["MainCharacter"]
	    )
        
        floor2Image = skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/frame_1541678261.9190319.png")
        self.floor2 = serpent.cv.extract_region_from_image(
                 floor2Image,
                 self.game.screen_regions["FloorDesc"]
	    )           
        floor3Image = skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/frame_1541616861.6503916.png")
        self.floor3 = serpent.cv.extract_region_from_image(
                 floor3Image,
                 self.game.screen_regions["FloorDesc"]
	    )
                  
        defeatImage= skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/frame_1541615338.5056944.png")
        self.defIm = serpent.cv.extract_region_from_image(
                 defeatImage,
                 self.game.screen_regions["Defeated"]
	    )
        descImage= skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/frame_1541432925.837391.png")
        self.des = serpent.cv.extract_region_from_image(
                 descImage,
                 self.game.screen_regions["DES"]
	    )
        ascImage= skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/frame_1541432927.8313663.png")
        self.asc = serpent.cv.extract_region_from_image(
                 ascImage,
                 self.game.screen_regions["ASC"]
	    )            
        #testTesseract= skimage.io.imread("R:/TFG/SerpentAI/datasets/collect_frames/example_01.png")
        main_player_model_file_path = "datasets/main_player_pixel_dungeon_dqn_0_1_.h5".format("/",os.sep)
        buddy_player_model_file_path = "datasets/buddy_player_pixel_dungeon_dqn_0_1_.h5".format("/",os.sep)        
        self.FirstEnemyKilled = False
        self.FirstDescent = False
        self.currentFloor = 1
        self.isDescending = False
        self.currentHP = 100
        self.previousHP = 100
        self.descended = False
        self.isFalling = False
        self.actualStep = 0

        self.dqn_main_player = BDQN(
        	model_file_path = main_player_model_file_path,
          input_shape = (100,100,4),
        	input_mapping = input_mapping,
        	action_space = movement_action_space,
        	replay_memory_size = 5000,
        	max_steps = 10000,
        	observe_steps = 0,
        	batch_size = 32,
        	initial_epsilon = 1,
        	final_epsilon = 0.01
          
        )


        self.dqn_buddy_player = BDQN(
        	model_file_path = buddy_player_model_file_path,
        	input_shape = (100,100,4),
        	input_mapping = input_mapping,
        	action_space = movement_action_space_buddy,
        	replay_memory_size = 5000,
        	max_steps = 10000,
        	observe_steps = 0,
        	batch_size = 32,
        	initial_epsilon = 1,
        	final_epsilon = 0.01
        )
        #self.dqn_main_player.enter_train_mode()
        #print(descIm)
        '''
        self.deadIm = serpent.cv.extract_region_from_image(
                 game_overImage,
                 self.game.screen_regions["GameOverDead"]
	     )       
        '''
    def handle_play(self, game_frame):
        self.isDescending = self.ascendDescend(game_frame)
        self.currentHP = self.computeActualHP(game_frame)
        self.falling(game_frame)
        

        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )
        
        if self.dqn_main_player.frame_stack is None:
            pipeline_game_frame = FrameGrabber.get_frames(
                [0],
                frame_shape=(100, 100),
                frame_type="PIPELINE",
                dtype="float64"
            ).frames[0]
            
            self.dqn_main_player.build_frame_stack(pipeline_game_frame.frame)
            self.dqn_buddy_player.frame_stack = self.dqn_main_player.frame_stack
            
        else:
            game_frame_buffer = FrameGrabber.get_frames(
                [0, 4, 8, 12],
                frame_shape=(100, 100),
                frame_type="PIPELINE",
                dtype="float64"
            )
        
        reward = self.calculate_reward()
        if self.dqn_main_player.mode == "TRAIN":
            
            self.game_state["run_reward_main"] += reward
            self.game_state["run_reward_buddy"] += reward
            
            self.dqn_main_player.append_to_replay_memory(
                    game_frame_buffer,
                    reward,
                    terminal= self.currentHP == 0 
            )
            
            self.dqn_buddy_player.append_to_replay_memory(
                    game_frame_buffer,
                    reward,
                    terminal= self.currentHP == 0 
            )            
                
            # Every 2000 steps, save latest weights to disk
            if self.dqn_main_player.current_step % 2000 == 0:
                    self.dqn_main_player.save_model_weights(
                        file_path_prefix= "datasets/dqn/dqn_main/"
                    )
                    self.dqn_buddy_player.save_model_weights(
                        file_path_prefix=f"datasets/dqn/dqn_buddy/"    
                    )

            # Every 20000 steps, save weights checkpoint to disk
            if self.dqn_main_player.current_step % 20000 == 0:
                    self.dqn_main_player.save_model_weights(
                        file_path_prefix= "datasets/dqn/dqn_main/",
                        is_checkpoint=True
                    )
                    self.dqn_buddy_player.save_model_weights(
                        file_path_prefix= "datasets/dqn/dqn_buddy/",
                        is_checkpoint=True
                    )                    

        elif self.dqn_main_player.mode == "RUN":
            self.dqn_main_player.update_frame_stack(game_frame_buffer)
            self.dqn_buddy_player.update_frame_stack(game_frame_buffer)

        run_time = datetime.now() - self.started_at

        serpent.utilities.clear_terminal()

        print(f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
        print("")

        print("MAIN NEURAL NETWORK:\n")
        self.dqn_main_player.output_step_data()

        print("")
        
        print("BUDDY NEURAL NETWORK:\n")
        self.dqn_buddy_player.output_step_data()

        print("")        
        print(f"CURRENT RUN: {self.game_state['current_run']}")
        print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward_main'] + self.game_state['run_reward_buddy'] , 2)}")
        print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
        print(f"CURRENT HEALTH: {self.currentHP}")

        print("")
        print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")

        print("")
        print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds (Run {self.game_state['record_time_alive'].get('run')}, {'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'} ")
        print("")

        print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")


        
        if self.currentHP <= 0:
            serpent.utilities.clear_terminal()
            timestamp = datetime.utcnow()

            gc.enable()
            gc.collect()
            gc.disable()

            timestamp_delta = timestamp - self.game_state["run_timestamp"]
            self.game_state["last_run_duration"] = timestamp_delta.seconds

            if self.dqn_main_player.mode in ["TRAIN","RUN"]:
                #Check for Records
                if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                    self.game_state["record_time_alive"] = {
                        "value": self.game_state["last_run_duration"],
                        "run": self.game_state["current_run"],
                        "predicted": self.dqn_main_player.mode == "RUN"
                }

            self.game_state["current_run_steps"] = 0

            self.input_controller.handle_keys([])

            if self.dqn_main_player.mode == "TRAIN":
                for i in range(16):
                    serpent.utilities.clear_terminal()
                    print(f"TRAINING ON MINI-BATCHES: {i + 1}/16")
                    print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                    self.dqn_main_player.train_on_mini_batch()
                    self.dqn_buddy_player.train_on_mini_batch()

            self.game_state["run_timestamp"] = datetime.utcnow()
            self.game_state["current_run"] += 1
            self.game_state["run_reward_main"] = 0
            self.game_state["run_reward_buddy"] = 0
            self.game_state["run_predicted_actions"] = 0
            
            self.restartLevel()
            
            if self.dqn_main_player.mode in ["TRAIN", "RUN"]:
                if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                    self.dqn_main_player.update_target_model()
                    self.dqn_buddy_player.update_target_model()
                    

                if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                    self.dqn_main_player.enter_run_mode()
                    self.dqn_buddy_player.enter_run_mode()
                else:
                    self.dqn_main_player.enter_train_mode()
                    self.dqn_buddy_player.enter_train_mode()

            return None
        if(self.actualStep%2 == 0):
            self.dqn_main_player.pick_action()
            self.dqn_main_player.generate_action()
            movement_keys = self.dqn_main_player.get_input_values()
            print("")
            print(" + ".join(list(map(lambda k: self.key_mapping.get(k.name), movement_keys))))
            self.input_controller.handle_keys(movement_keys)
            self.dqn_main_player.erode_epsilon(factor=2)
            self.dqn_main_player.next_step()
            #time.sleep(1)	        
        else:
            self.dqn_buddy_player.pick_action()
            self.dqn_buddy_player.generate_action()
            movement_keys_buddy = self.dqn_buddy_player.get_input_values()         
            print("")
            print(" + ".join(list(map(lambda k: self.key_mapping.get(k.name),movement_keys_buddy))))
            self.input_controller.handle_keys(movement_keys_buddy)
            self.dqn_buddy_player.erode_epsilon(factor=2)
            self.dqn_buddy_player.next_step()
            #time.sleep(1)
        #movement_keys = self.dqn_main_player.get_input_values()
        #movement_keys_buddy = self.dqn_buddy_player.get_input_values()        

        #print("")
        #print(" + ".join(list(map(lambda k: self.key_mapping.get(k.name), movement_keys + movement_keys_buddy))))
        #self.input_controller.handle_keys(movement_keys + movement_keys_buddy)

        if self.dqn_main_player.current_action_type == "PREDICTED":
            self.game_state["run_predicted_actions"] += 1

        self.game_state["current_run_steps"] += 1
        self.actualStep += 1      

        

        

    def _reset_game_state(self):
        self.game_state = {
            "current_run": 1,
            "current_run_steps": 0,
            "run_reward_main": 0,
            "run_reward_buddy": 0,
            "run_future_rewards": 0,
            "run_predicted_actions": 0,
            "run_timestamp": datetime.utcnow(),
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list()
        }

    def restartLevel(self):
        print("I'm dead so i create a new game")
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(1)
        self.input_controller.click_screen_region(
             button=MouseButton.LEFT,
             screen_region="StartNewGameOver"
        )
        time.sleep(2)
        self.isClicking = True
        self.input_controller.click_screen_region(
             button=MouseButton.LEFT,
             screen_region="StartNewGame"
        )
        time.sleep(3)
        self.currentFloor = 1 
        self.currentHP = 100
        self.previousHP = 100
    

    def falling(self,game_frame):
        self.chasm = serpent.cv.extract_region_from_image(
                 game_frame.frame,
                 self.game.screen_regions["Chasm"]
	     )
        text = pytesseract.image_to_string(self.chasm)
        if text == "CHBSITI":
            self.isFalling = True
            self.input_controller.click_screen_region(
                    button=MouseButton.LEFT,
                    screen_region="NoChasm"
            )
            time.sleep(2)
            
    def defeatEnemy(self,game_frame):
        self.defeat = serpent.cv.extract_region_from_image(
                 game_frame.frame,
                 self.game.screen_regions["Defeated"]
	     )
        text = pytesseract.image_to_string(self.defeat)
        print(text)                
    
    def ascendDescend(self,game_frame):
        self.des = serpent.cv.extract_region_from_image(
                 game_frame.frame,
                 self.game.screen_regions["DES"]
	     )
        self.asc = serpent.cv.extract_region_from_image(
                 game_frame.frame,
                 self.game.screen_regions["ASC"]
	     )            
        text = pytesseract.image_to_string(self.des)
        text2 = pytesseract.image_to_string(self.asc)
        if text == "Descending" or text == "Descending.":
            self.currentFloor += 1 
            print("Congratulations you descended to level " + str(self.currentFloor))
            time.sleep(2)
            self.descended = True
            return True
        if text2 == "ascending" or text2 == "jascending" or text2 == "ascending.":
            self.currentFloor -= 1 
            print("You returned to level " + str(self.currentFloor))
            time.sleep(2)
            return True
        return False
    
    
    def enemyDefeated(self,game_frame):
        auxDef = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["Defeated"]
	     )

        aux = [0,0,0]
        n = len(auxDef)
        m = len(auxDef[0])
        for i in range(0,n):
        	for j in range(0,m):
        		aux[0] = aux[0] + abs(auxDef[i][j][0] - self.defIm[i][j][0])
        		aux[1] = aux[1] + abs(auxDef[i][j][1] - self.defIm[i][j][1])
        		aux[2] = aux[2] + abs(auxDef[i][j][2] - self.defIm[i][j][2])

        aux[0] = aux[0]/(n*m)
        aux[1] = aux[1]/(n*m)
        aux[2] = aux[2]/(n*m)

        if self.FirstEnemyKilled:
            currentEnemyKilled = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["EnemyKilled"]
            )
            aux2 = [0,0,0]
            n2 = len(currentEnemyKilled)
            m2 = len(currentEnemyKilled[0])
            for i in range(0,n2):
                for j in range(0,m2):
                    aux2[0] = aux2[0] + abs(currentEnemyKilled[i][j][0] - self.currentEnemyKilled[i][j][0])
                    aux2[1] = aux2[1] + abs(currentEnemyKilled[i][j][1] - self.currentEnemyKilled[i][j][1])
                    aux2[2] = aux2[2] + abs(currentEnemyKilled[i][j][2] - self.currentEnemyKilled[i][j][2])

            aux2[0] = aux2[0]/(n2*m2)
            aux2[1] = aux2[1]/(n2*m2)
            aux2[2] = aux2[2]/(n2*m2)
            print("Diff",aux2)            
        #print("Pos esta es la diferencia lok0",aux)
        if aux[0] < 15 and aux[1] < 15 and aux[2] < 15 :
            print("Diferencia por color: ", aux)
            self.FirstEnemyKilled = True
            self.currentEnemyKilled = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["EnemyKilled"]
            )            
            return True
        return False
    
    
    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def computeActualHP(self,game_frame):
    	 
        game_area_1 = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["FirstFractionHP"]
        )
        game_area_2 = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["SecondFractionHP"]
        )
        game_area_3 = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["ThirdFractionHP"]
        )                
        game_area_4 = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["FourthFractionHP"]
        )
        game_area_5 = serpent.cv.extract_region_from_image(
                game_frame.frame,
                self.game.screen_regions["FifthFractionHP"]
        )        
        #print(game_area_1[0][0][0])
        hp = 0
        #Max hp is 100 and min is 0
        aux = [0,0,0]
        n = len(game_area_1)
        m = len(game_area_1[0])
        for i in range(0,n):
        	for j in range(0,m):
        		aux[0] = aux[0] + game_area_1[i][j][0]

        aux[0] = aux[0]/(n*m)
        #print("Primera region color medio", aux[0])
        if aux[0] > 100:
        	hp += 20
        elif aux[0] > 80:
        	hp += 10

        aux = [0,0,0]
        n = len(game_area_2)
        m = len(game_area_2[0])
        for i in range(0,n):
        	for j in range(0,m):
        		aux[0] = aux[0] + game_area_2[i][j][0]

        aux[0] = aux[0]/(n*m)
        #print("Segunda region color medio", aux[0])
        if aux[0] > 100:
        	hp += 20
        elif aux[0] > 80:
        	hp += 10	

        aux = [0,0,0]
        n = len(game_area_3)
        m = len(game_area_3[0])
        for i in range(0,n):
        	for j in range(0,m):
        		aux[0] = aux[0] + game_area_3[i][j][0]

        aux[0] = aux[0]/(n*m)
        #print("Tercera region color medio", aux[0])
        if aux[0] > 100:
        	hp += 20        
        elif aux[0] > 80:
        	hp += 10
        		
        aux = [0,0,0]
        n = len(game_area_4)
        m = len(game_area_4[0])
        for i in range(0,n):
        	for j in range(0,m):
        		aux[0] = aux[0] + game_area_4[i][j][0]
	
        aux[0] = aux[0]/(n*m)
        #print("Cuarta region color medio", aux[0])
        if aux[0] > 100:
        	hp += 20
        elif aux[0] > 80:
        	hp += 10

        aux = [0,0,0]
        n = len(game_area_5)
        m = len(game_area_5[0])
        for i in range(0,n):
        	for j in range(0,m):
        		aux[0] = aux[0] + game_area_5[i][j][0]

        aux[0] = aux[0]/(n*m)
        #print("Quinta region color medio", aux[0])
        if aux[0] > 100:
        	hp += 20
        elif aux[0] > 80:
        	hp += 10	 
        #print("Actual health points: ",hp)
        self.previousHP = self.currentHP
        self.currentHP = hp
        return hp
    
    
    def calculate_reward(self):
        reward = 0
        
        reward += (-1 if self.currentHP < self.previousHP else 0)
        reward += (10 if self.descended else 0)
        reward += (-0.1 if self.isFalling else 0)
        if self.descended:
            self.descended = False
            
        if self.isFalling:
            self.isFalling = False
            
        return reward        