from serpent.game import Game

from .api.api import FineSweeperAPI

from serpent.utilities import Singleton




class SerpentFineSweeperGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"

        kwargs["window_name"] = "Fine Sweeper"

        kwargs["app_id"] = "367000"
        kwargs["app_args"] = None
        
        
        

        super().__init__(**kwargs)

        self.api_class = FineSweeperAPI
        self.api_instance = None

    @property
    def screen_regions(self):
        regions = {
            "SAMPLE_REGION": (0, 0, 0, 0)
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
