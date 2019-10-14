# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from constants import CONSTANTS
from leduc import Leduc


class CFR:
    """
    initialized with the Leduc game tree. provides functionality of solving the strategy,
    as well as demonstration of the solving process
    """

    def __init__(self):

        # search tree
        self.tree = Leduc()

        self.strategy = {}

        # distance record along the iterations
        self.strategy_distances = []

    def solve(self, num_iters=15000, target_distance=1e-5, show_process=False):
        """
        solve the strategy
        :param target_distance: when the distance is smaller than this value, stop solving and return the strategy
        :param show_process: show the solving process
        :param num_iters: time of iterations
        :return: the solved strategy
        """
        regrets = {}

        action_counts = {}

        # strategy_t holds the strategy at time t;
        # strategy_t_1 holds the strategy at time t + 1
        strategy_t = {}
        strategy_t_1 = {}

        average_strategy = None
        average_strategy_snapshot = None

        for t_iter in range(num_iters):
            for i in [CONSTANTS.players2indexes['p1'], CONSTANTS.players2indexes['p2']]:
                self._solve_recursive(self.tree, self.tree.root, i, t_iter, 1.0, 1.0, regrets,
                                      action_counts, strategy_t, strategy_t_1)

            # every 100 iterations, we compute a new average strategy and compare it with the former one
            if (t_iter % 100 == 0) and (average_strategy is not None):
                if show_process:
                    print("t: {}".format(t_iter))
                if average_strategy_snapshot is not None:
                    distance = CFR._compute_strategy_distance(average_strategy, average_strategy_snapshot)
                    self.strategy_distances.append(distance)
                    if show_process:
                        print("distance: {}".format(distance))
                    # distance small enough, return
                    if distance < target_distance:
                        return average_strategy

                average_strategy_snapshot = average_strategy.copy()

            average_strategy = CFR._compute_average_strategy(action_counts)
            strategy_t = strategy_t_1.copy()

        return average_strategy

    @staticmethod
    def _solve_recursive(tree, node, i, t_iter, pp1, pp2, regrets, action_counts,
                         strategy_t, strategy_t_1):
        """
        recursively solve the strategy
        :param tree: the game tree
        :param node: the current node
        :param i: player index
        :param t_iter: which iteration
        :param pp1: prior probability of player 1
        :param pp2: prior probability of player 2
        :param regrets: regrets
        :param action_counts: dict, with keys the information sets and values another dict from action to action counts
        :param strategy_t: current strategy at iter t
        :param strategy_t_1: next strategy
        :return: value
        """
        # if the node is terminal, then return the utility for the player
        if node.is_terminal():
            return node.utility[i]
        # if the next node is a chance node, then sample one chance action
        elif node.which_player() == CONSTANTS.players2indexes['chance']:
            action = node.sample_chance_action()
            return CFR._solve_recursive(tree, node.children[action], i, t_iter,
                                        pp1, pp2, regrets, action_counts, strategy_t, strategy_t_1)

        # retrieve the information set
        information_set = tree.retrieve_information_sets()[node]

        # initialize values
        player = node.which_player()
        value = 0
        available_actions = node.available_actions()
        values = {a: 0 for a in available_actions}

        # initialize strategy_t
        if information_set not in strategy_t:
            strategy_t[information_set] = {a: 1.0 / float(len(available_actions))
                                           for a in available_actions}

        # compute the counterfactual values for the information set
        for a in available_actions:
            if player == CONSTANTS.players2indexes['p1']:
                values[a] = CFR._solve_recursive(tree, node.children[a], i, t_iter,
                                                 strategy_t[information_set][a] * pp1, pp2,
                                                 regrets, action_counts, strategy_t, strategy_t_1)
            else:
                values[a] = CFR._solve_recursive(tree, node.children[a], i, t_iter,
                                                 pp1, strategy_t[information_set][a] * pp2,
                                                 regrets, action_counts, strategy_t, strategy_t_1)
            value += strategy_t[information_set][a] * values[a]

        # compute the regrets
        if information_set not in regrets:
            regrets[information_set] = {a: 0.0 for a in available_actions}
        if player == i:
            for a in available_actions:
                pi_minus_i = pp1 if i == 2 else pp2
                pi_i = pp1 if i == 1 else pp2
                regrets[information_set][a] += (values[a] - value) * pi_minus_i
                if information_set not in action_counts:
                    action_counts[information_set] = {
                        ad: 0.0 for ad in available_actions}
                action_counts[information_set][a] += pi_i * \
                                                     strategy_t[information_set][a]

            # update strategy_t_1
            strategy_t_1[information_set] = CFR._regret_matching(regrets[information_set])

        return value

    @staticmethod
    def _regret_matching(regrets):
        """
        compute the new strategy according to the regrets
        :param regrets:
        :return: strategy after regret matching
        """

        # if there is no positive regret, return the uniform strategy
        if max([v for k, v in regrets.items()]) <= 0.0:
            return {a: 1.0 / float(len(regrets)) for a in regrets}
        else:
            denominator = sum([max(0.0, v) for k, v in regrets.items()])
            return {k: max(0.0, v) / denominator for k, v in regrets.items()}

    @staticmethod
    def _compute_average_strategy(action_counts):
        average_strategy = {}
        for information_set in action_counts:
            num_actions = sum([v for k, v in action_counts[information_set].items()])
            if num_actions > 0:
                average_strategy[information_set] = {k: float(v) / float(num_actions)
                                                     for k, v in action_counts[information_set].items()}

        return average_strategy

    @staticmethod
    def _compute_strategy_distance(s1, s2):
        """
        given two strategies, compute the euler distance between them
        :param s1: strategy 1
        :param s2: strategy 2
        :return: distance
        """
        common_keys = [k for k in s1.keys() if k in s2.keys()]
        distances = []
        for information_set in common_keys:
            prob_dist_diff = [float(s1[information_set][a] - s2[information_set][a]) ** 2
                              for a in s1[information_set]]
            distances.append(np.sqrt(np.mean(prob_dist_diff)))

        return np.mean(distances)

    def retrieve_distance_record(self):
        return self.strategy_distances


def plot_distances(d):
    x = np.array(range(len(distances))) * 100 + 100
    plt.figure(figsize=(18, 10), dpi=300)
    plt.title("Distances between the latest and the last strategy")
    plt.xlabel("Iterations")
    plt.ylabel("Euler Distance")
    plt.plot(x, d)
    plt.savefig('distances')


if __name__ == "__main__":
    cfr = CFR()

    strategy = cfr.solve(show_process=True)

    distances = cfr.retrieve_distance_record()

    plot_distances(distances)
