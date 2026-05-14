import numpy as np
import torch

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def win_rate(env, dfops, terminal, red_win_num, blue_win_num, draw_num, red_fall_num, blue_fall_num, red_harm, blue_harm):
    # 统计胜率
    if terminal == 0:
        if env.world.fighters[0].combat_data.bloods > env.world.fighters[1].combat_data.bloods:
            red_win_num += 1
            dfops.record_episode_result(dfops.opponent_id, 1)
        elif env.world.fighters[0].combat_data.bloods < env.world.fighters[1].combat_data.bloods:
            blue_win_num += 1
            dfops.record_episode_result(dfops.opponent_id, 2)
        else:
            draw_num += 1
            dfops.record_episode_result(dfops.opponent_id, 0)

    if terminal == 1:
        red_win_num += 1
        dfops.record_episode_result(dfops.opponent_id, 1)

    if terminal == 2:
        blue_win_num += 1
        dfops.record_episode_result(dfops.opponent_id, 2)

    if terminal == 3:
        draw_num += 1
        dfops.record_episode_result(dfops.opponent_id, 0)

    if terminal == 4:
        red_fall_num += 1
        blue_win_num += 1
        dfops.record_episode_result(dfops.opponent_id, 2)

    if terminal == 5:
        blue_fall_num += 1
        red_win_num += 1
        dfops.record_episode_result(dfops.opponent_id, 1)
    red_harm += (3 - env.world.fighters[1].combat_data.bloods)
    blue_harm += (3 - env.world.fighters[0].combat_data.bloods)
    return red_win_num, blue_win_num, draw_num, red_fall_num, blue_fall_num, red_harm, blue_harm

