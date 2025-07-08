import math

from chiron_utils.utils import return_logger
from conf.agents_pb2 import *
from diplomacy import Game as MilaGame

from fairdiplomacy.agents.model_sampled_agent import ModelSampledAgent
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import (
    Timestamp,
)
import heyhi

logger = return_logger(__name__)

DEFAULT_DEADLINE = 5


class BlueprintWrapper:

    def __init__(self, power_name, game):
        self.game: MilaGame = game
        self.dipcc_game: Game = None

        agent_config = heyhi.load_config('/home/ubuntu/github/diplomacy_searchbot/conf/common/agents/model_sampled.prototxt')
        self.agent = ModelSampledAgent(agent_config.model_sampled)

        self.power_name = power_name
        self.dipcc_game = self.start_dipcc_game(power_name)
        self.dipcc_current_phase = self.dipcc_game.get_current_phase()
    
    def get_orders(self, game):
        # update dipcc game state
        self.game = game
        if self.has_phase_changed():
            self.update_and_process_dipcc_game()
            self.dipcc_current_phase = self.game.get_current_phase()

        # fix issue that there is a chance where retreat phase appears in dipcc but not mila 
        while self.dipcc_game and self.has_phase_changed():
            agent_orders = self.agent.get_orders(self.dipcc_game, self.power_name)
            self.dipcc_game.set_orders(self.power_name, agent_orders)
            self.dipcc_game.process()
            self.dipcc_current_phase = self.dipcc_game.get_current_phase()

        # get orders
        orders = self.suggest_move(self.power_name)
        return orders
    
    def suggest_move(self, power_name):
        agent_orders = list(self.agent.get_orders(self.dipcc_game, power_name))
        return agent_orders
 
    def has_phase_changed(self)->bool:
        """ 
        check game phase 
        """
        return self.dipcc_game.get_current_phase() != self.game.get_current_phase()

    def update_and_process_dipcc_game(self):
        """     
        Inputs orders from the bygone phase into the dipcc game and process the dipcc game.
        """

        dipcc_game = self.dipcc_game
        mila_game = self.game
        dipcc_phase = dipcc_game.get_state()['name'] # short name for phase
        if dipcc_phase in mila_game.order_history:
            orders_from_prev_phase = mila_game.order_history[dipcc_phase] 
            
            # gathering orders from other powers from the phase that just ended
            for power, orders in orders_from_prev_phase.items():
                dipcc_game.set_orders(power, orders)

        dipcc_game.process() # processing the orders set and moving on to the next phase of the dipcc game

    def start_dipcc_game(self, power_name: POWERS) -> Game:

        deadline = self.game.deadline

        if deadline ==0:
            deadline = DEFAULT_DEADLINE
        else:
            deadline = int(math.ceil(deadline/60))
        
        game = Game()

        game.set_scoring_system(Game.SCORING_SOS)
        game.set_metadata("phase_minutes", str(deadline))

        while game.get_state()['name'] != self.game.get_current_phase():
            self.update_past_phase(game,  game.get_state()['name'], power_name)

        return game
    
    def update_past_phase(self, dipcc_game: Game, phase: str, power_name: str):
        if phase not in self.game.message_history:
            dipcc_game.process()
            return

        phase_message = self.game.message_history[phase]
        for timesent, message in phase_message.items():

            if message.recipient != power_name or message.sender != power_name:
                continue

            dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)

            if message.recipient not in self.game.powers:
                continue

            dipcc_game.add_message(
                message.sender,
                message.recipient,
                message.message,
                time_sent=dipcc_timesent,
                increment_on_collision=True,
            )

        phase_order = self.game.order_history[phase] 

        for power, orders in phase_order.items():
            dipcc_game.set_orders(power, orders)
        
        dipcc_game.process()
