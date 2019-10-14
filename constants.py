# coding=utf-8


class Constants:
    def __init__(self):
        # >>> 游戏基本参数 - Game facts
        # 第一轮加注额 -- first round raise
        self.FIRST_ROUND_RAISE = 2

        # 第二轮加注额 -- second round raise
        self.SECOND_ROUND_RAISE = 4

        # 牌 -- deck
        # 6 cards, Jack (10th), Queen (11th), King (12th), each has 2 cards.
        self.DECK = 2 * [10, 11, 12]

        # 玩家序号 -- Indexes of players
        # terminal: terminal node
        # chance: chance player (or dealer), the player who deals cards
        # p1: player 1
        # p2: player 2
        self.players2indexes = {'terminal': -1, 'chance': 0, 'p1': 1, 'p2': 2}

        # 操作序号 -- Indexes of actions
        self.actions2indexes = {'fold': 0, 'call/check': 1, 'bet/raise': 2}
        # <<< 游戏基本参数 - Game facts


CONSTANTS = Constants()
