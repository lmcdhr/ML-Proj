# coding: utf-8

import numpy as np
from constants import CONSTANTS


class Node:
    """
    Node of the search tree (or search space). Can generate local search tree from the current node.
    The depth of the tree
    """

    def __init__(self, p):
        """
        :param p: The player who took the action that led to this node
        """
        # >>> 游戏信息 - Game information
        self.player_index = p
        # <<< 游戏信息 - Game information

        # >>> 搜索树 - Search tree
        # children is a dictionary, with keys being the actions that leads to the child nodes,
        # and values being the child nodes
        self.children = {}

        # players in the hidden_from list are not able to reach this node
        self.hidden_from = []

        # utilities used in terminal nodes, with keys being the player names and values being
        # the utility
        self.utility = {}

        # if the node is a chance node (dealing or revealing cards), the following element
        # records the probability of different chance actions
        self.chance_probabilities = {}
        # <<< 搜索树 - Search tree

    def is_terminal(self):
        return self.player_index == CONSTANTS.players2indexes['terminal']

    def sample_chance_action(self):
        """
        samples a chance action, only valid for chance nodes
        :return:
        """
        # check if it is a chance node
        assert(self.player_index == CONSTANTS.players2indexes['chance'])
        actions = [a for a in self.chance_probabilities]
        probabilities = [p for a, p in self.chance_probabilities.items()]
        return np.random.choice(actions, p=probabilities)

    def which_player(self):
        return self.player_index

    def available_actions(self):
        return [action for action in self.children.keys()]


class Leduc:
    def __init__(self):
        # root node of the search tree
        self.root = Leduc._generate_search_tree_recursive()

        self.info_set = {}
        # generate info sets
        self._generate_info_sets()

    # >>> generate search tree
    @staticmethod
    def _generate_search_tree_recursive(action_route=[], deck=CONSTANTS.DECK):
        """
        recursively generate the search tree
        :param deck: deck, initially the whole deck
        :param action_route: the action history route starting from root to the current node
        :return: root node, with the subtrees generated
        """
        # root, create chance nodes for p1
        if len(action_route) == 0:
            root = Node(CONSTANTS.players2indexes['chance'])
            # p2 is hidden from this node
            root.hidden_from = [CONSTANTS.players2indexes['p2']]
            for card in deck:
                # generate a game tree
                if card not in root.children:
                    deck_remaining = deck.copy()
                    deck_remaining.remove(card)
                    root.children[card] = Leduc._generate_search_tree_recursive(action_route + [card], deck_remaining)
                    root.chance_probabilities[card] = 1.0 / float(len(deck))
                else:
                    root.chance_probabilities[card] += 1.0 / float(len(deck))
            return root
        # create chance nodes for p2
        elif len(action_route) == 1:
            node = Node(CONSTANTS.players2indexes['chance'])
            # p1 is hidden from this node
            node.hidden_from = [CONSTANTS.players2indexes['p1']]
            for card in deck:
                if card not in node.children:
                    deck_remaining = deck.copy()
                    deck_remaining.remove(card)
                    node.children[card] = Leduc._generate_search_tree_recursive(action_route + [card], deck_remaining)
                    node.chance_probabilities[card] = 1.0 / float(len(deck))
                else:
                    node.chance_probabilities[card] += 1.0 / float(len(deck))
            return node

        # actions in the current round
        current_round_actions = []
        # list of rounds, at most two rounds
        rounds = []
        public_cards = []
        # first two actions are dealing of the private cards
        for a in action_route[2:]:
            # a < 10, an action
            if a < 10:
                current_round_actions.append(a)
            # otherwise, a card
            else:
                public_cards.append(a)
                # a chance node indicates the end of a round, so add to the rounds list
                if len(current_round_actions) > 0:
                    rounds.append(current_round_actions)
                    current_round_actions = []
        if len(current_round_actions) > 0:
            rounds.append(current_round_actions)

        # check the length of rounds, which indicates where is the current round
        # at most two rounds
        assert len(rounds) <= 2
        # check if it is the end of the round
        if len(rounds) == 0:
            current_round_actions = []
        else:
            current_round_actions = rounds[-1]
        # here we don't use CONSTANTS because that makes the code too long
        # fold: 0, check/call: 1, bet/raise: 2
        if (current_round_actions == [1, 1] or current_round_actions[-2:] == [2, 1] or
                current_round_actions[-2:] == [2, 0]):

            # round 1, create chance nodes
            if len(rounds) == 1:
                if len(public_cards) > 0:
                    # already created the chance node. create node for p2 (p2 acts first
                    # in the second round)
                    node = Node(CONSTANTS.players2indexes['p2'])
                    node.children[1] = Leduc._generate_search_tree_recursive(action_route + [1], deck)
                    node.children[2] = Leduc._generate_search_tree_recursive(action_route + [2], deck)
                    return node
                else:
                    # create chance node
                    node = Node(CONSTANTS.players2indexes['chance'])
                    for card in deck:
                        if card not in node.children:
                            deck_remaining = deck.copy()
                            deck_remaining.remove(card)
                            node.children[card] = Leduc._generate_search_tree_recursive(action_route + [card],
                                                                                        deck_remaining)
                            node.chance_probabilities[card] = 1.0 / float(len(deck))
                        else:
                            node.chance_probabilities[card] += 1.0 / float(len(deck))

                    return node
            # end of the game, compute utility
            else:
                node = Node(CONSTANTS.players2indexes['terminal'])
                node.utility = Leduc.compute_utility(action_route)
                return node
        else:
            # find out whose turn it is
            player = CONSTANTS.players2indexes['p1'] if len(current_round_actions) % 2 == 0 \
                else CONSTANTS.players2indexes['p2']
            node = Node(player)

            # available actions
            available_actions = []
            if not current_round_actions:
                available_actions = [1, 2]
            elif current_round_actions == [2]:
                available_actions = [0, 1, 2]
            elif current_round_actions == [2, 2]:
                available_actions = [0, 1, 2]
            elif current_round_actions == [2, 2, 2]:
                available_actions = [0, 1, 2]
            elif current_round_actions == [2, 2, 2, 2]:
                available_actions = [0, 1]
            elif current_round_actions == [1]:
                available_actions = [1, 2]
            elif current_round_actions == [1, 2]:
                available_actions = [0, 1, 2]
            elif current_round_actions == [1, 2]:
                available_actions = [0, 1, 2]
            elif current_round_actions == [1, 2, 2]:
                available_actions = [0, 1, 2]
            elif current_round_actions == [1, 2, 2, 2]:
                available_actions = [0, 1, 2]
            elif current_round_actions == [1, 2, 2, 2, 2]:
                available_actions = [0, 1]

            for action in available_actions:
                node.children[action] = Leduc._generate_search_tree_recursive(action_route + [action], deck)

            return node
    # <<< generate search tree

    # >>> information sets
    def _generate_info_sets(self):
        info_set_p1 = self._generate_info_sets_player(CONSTANTS.players2indexes['p1'])
        info_set_p2 = self._generate_info_sets_player(CONSTANTS.players2indexes['p2'])
        for k, v in info_set_p1.items():
            if k.player_index == CONSTANTS.players2indexes['p1']:
                self.info_set[k] = v
        for k, v in info_set_p2.items():
            if k.player_index == CONSTANTS.players2indexes['p2']:
                self.info_set[k] = v

    def _generate_info_sets_player(self, player):
        """

        :param player:
        :return:
        """

        info_set = dict()

        node_stack = [self.root]
        action_routes_stack = [[]]

        while len(node_stack) > 0:
            node = node_stack.pop()
            action_route = action_routes_stack.pop()

            info_set[node] = tuple(action_route)

            for action, child in node.children.items():
                node_stack.append(child)

                if player in node.hidden_from:
                    action_routes_stack.append(action_route + [-1])
                else:
                    action_routes_stack.append(action_route + [action])

        return info_set

    # <<< information sets

    # >>> tree demonstration
    def print_leaves(self):
        Leduc._print_tree_recursive(self.root, [])

    @staticmethod
    def _print_tree_recursive(node, actions):
        """
        recursively print the nodes information and the action route that leads to the nodes
        :param node: start node
        :param actions: action route that leads to the node
        :return:
        """
        if not node.children:
            print(actions, node.utility)
        for action, child in node.children.items():
            Leduc._print_tree_recursive(child, actions + [action])
    # <<< tree demonstration

    @staticmethod
    def compute_antes(action_route):
        """
        given the action route, compute the current antes of the players
        :param action_route: a list of actions starting from the private cards to the current state
        :return: a dictionary of the antes, with keys being the player index
        """
        antes = {CONSTANTS.players2indexes['p1']: 1, CONSTANTS.players2indexes['p2']: 1}
        # used for figuring out the other player
        the_other_player = {CONSTANTS.players2indexes['p1']: CONSTANTS.players2indexes['p2'],
                            CONSTANTS.players2indexes['p2']: CONSTANTS.players2indexes['p1']}

        # only dealt private cards
        if len(action_route) <= 2:
            return antes
        # no need to consider dealings
        action_route = action_route[2:]

        player = CONSTANTS.players2indexes['p1']

        raise_amount = CONSTANTS.FIRST_ROUND_RAISE

        for action in action_route:
            # revealing the public card. antes remain the same, round number increases
            if action >= 10:
                raise_amount = CONSTANTS.SECOND_ROUND_RAISE
                player = CONSTANTS.players2indexes['p2']
            elif action == CONSTANTS.actions2indexes['fold']:
                return antes
            elif action == CONSTANTS.actions2indexes['call/check']:
                antes[player] = antes[the_other_player[player]]
            elif action == CONSTANTS.actions2indexes['bet/raise']:
                antes[player] = antes[the_other_player[player]]
                antes[player] += raise_amount

            # switch side
            player = the_other_player[player]

        return antes

    @staticmethod
    def compute_utility(action_route):
        """
        given the action route, compute the utilities of players at the terminal node
        :param action_route:
        :return:
        """
        antes = Leduc.compute_antes(action_route)
        private_cards = {CONSTANTS.players2indexes['p1']: action_route[0],
                         CONSTANTS.players2indexes['p2']: action_route[1]}
        public_card = [a for a in action_route[2:] if a >= 10][0]

        if private_cards[CONSTANTS.players2indexes['p1']] == public_card:
            winner = CONSTANTS.players2indexes['p1']
        elif private_cards[CONSTANTS.players2indexes['p2']] == public_card:
            winner = CONSTANTS.players2indexes['p2']
        elif private_cards[CONSTANTS.players2indexes['p1']] > private_cards[CONSTANTS.players2indexes['p2']]:
            winner = CONSTANTS.players2indexes['p1']
        else:
            winner = CONSTANTS.players2indexes['p2']

        if winner == CONSTANTS.players2indexes['p2']:
            loser = CONSTANTS.players2indexes['p1']
        else:
            loser = CONSTANTS.players2indexes['p2']

        return {winner: antes[loser], loser: -antes[loser]}

    def retrieve_information_sets(self):
        return self.info_set


# testing
if __name__ == '__main__':
    leduc = Leduc()
    leduc.print_leaves()
