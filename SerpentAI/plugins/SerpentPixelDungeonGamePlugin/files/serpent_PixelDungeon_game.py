from serpent.game import Game

from .api.api import PixelDungeonAPI

from serpent.utilities import Singleton




class SerpentPixelDungeonGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["window_name"] = "Shattered Pixel Dungeon"

        
        
        kwargs["executable_path"] = "C://Users//Riqui//Desktop//shattered-pixel-dungeon-gdx-master//shattered-pixel-dungeon-gdx-master//desktop//build//libs//ShatteredPixelDungeon.exe"
        
        

        super().__init__(**kwargs)

        self.api_class = PixelDungeonAPI
        self.api_instance = None
        
        self.frame_transformation_pipeline_string = "RESIZE:100x100|GRAYSCALE|FLOAT"

        self.frame_width = 100
        self.frame_height = 100
        self.frame_channels = 0
        
    @property
    def screen_regions(self):
        regions = {
            "SAMPLE_REGION": (0, 0, 0, 0),
            "MainCharacter": (322, 212, 367, 243),
            "Level": (71, 68, 99, 97),
            "StartNewGameOver": (258, 214, 297, 273),
            "StartNewGame": (501, 193, 541, 248),
            "MegaDescending": (592, 39, 609, 201),
            "Descending": (593, 39, 607, 105),
            "FloorDesc": (592, 188, 608, 200),
            "DES": (329, 152, 361, 290),
            "ASC": (331, 161, 358, 282),
            "Chasm": (235, 51, 270, 133),
            "NoChasm": (411, 220, 430, 249),
            "Sentence": (588, 0, 612, 344),
            "Ascensing": (593, 40, 607, 90),
            "FloorAsc": (591, 169, 609, 189),
            "Descending2": (330, 153, 359, 212),
            "Defeated": (593, 41, 607, 110),
            "EnemyKilled": (590, 39, 611, 233),
            "Pick": (596, 40, 607, 113),
            "ObjectPicked": (592, 37, 611, 207),
            "GameOverDead": (211, 117, 285, 339),
            #From left to right, if first fraction is completly empty then you are dead
            "FirstFractionHP": (8, 89, 21, 121),
            "SecondFractionHP": (9, 120, 21, 151),
            "ThirdFractionHP": (9, 150, 21, 181),
            "FourthFractionHP": (9, 180, 21, 211),
            "FifthFractionHP": (8, 210, 21, 240)



        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
