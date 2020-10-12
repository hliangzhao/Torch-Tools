"""
This file defines the environment of job shop scheduling problem.
Each job is consists of several modules (the module number is the same as the machine number).
For each module of each job, the processing machine is fixed and non-repetitive.

A typical example:
    5 jobs, 4 machines:
    machines each module choose:
        [[1, 3, 0, 2],
         [0, 2, 1, 3],
         [3, 1, 2, 0],
         [1, 3, 0, 2],
         [0, 1, 2, 3]]
    processing time of each module:
        [[18, 20, 21, 17],
         [18, 26, 15, 16],
         [17, 18, 27, 23],
         [18, 21, 25, 15],
         [22, 29, 28, 21]]
"""
import numpy as np
import random
import tools
import matplotlib.pyplot as plt


JOB_PROCESS_TIME_MIN = 15
JOB_PROCESS_TIME_MAX = 30


class Env:
    def __init__(self, machine_num, job_num, job_process_time_min=JOB_PROCESS_TIME_MIN, job_process_time_max=JOB_PROCESS_TIME_MAX):
        self.machine_num = machine_num
        self.job_num = job_num
        self.job_process_time_min = job_process_time_min
        self.job_process_time_max = job_process_time_max

        self.jobs_process_time, self.jobs_process_order = self.generate_scenario()     # randomly generated
        self.starttime = np.zeros((self.machine_num, self.job_num))
        self.endtime = np.zeros((self.machine_num, self.job_num))
        self.feature_num = 2             # more features can be extracted!
        self.scheduling_plan = np.zeros((self.machine_num, self.job_num, 2), dtype=int)
        self.actions = []                # list of [job_id, job_progress]

    def generate_scenario(self):
        """
        Generate the processing time of each module of job's on the generated order of machines.
        """
        process_t, process_o = [], []
        for j in range(self.job_num):
            process_t.append(random.sample(range(self.job_process_time_min, self.job_process_time_max), self.machine_num))
            process_o.append(random.sample(range(self.machine_num), self.machine_num))
        return np.array(process_t), np.array(process_o)

    def get_jobs_progress(self):
        """
        Get jobs' processing progress according to the actions taken.
        """
        job_progresses = [0 for _ in range(self.job_num)]
        for job_id, job_progress in self.actions:
            if job_progress < self.machine_num - 1:
                # goto the next module (of this job)
                job_progresses[job_id] = job_progress + 1
            else:
                # job finished
                job_progresses[job_id] = -1
        return [[i, job_progresses[i]] for i in range(self.job_num)]

    def get_feature(self, job_id, job_progress):
        """
        Here feature is one kind of encoding of state of the given job.
        """
        machine_id = self.jobs_process_order[job_id, job_progress]

        if job_progress == -1:
            # job finished
            feature = [-1] * self.feature_num
        elif job_progress == 0:
            # job start
            feature = [0.5, 0.5]
        else:
            # job in processing
            machines_endtime = np.max(self.endtime, axis=1)
            job_cur_endtime = np.sum(self.jobs_process_time[job_id, :job_progress])
            job_overall_time = np.sum(self.jobs_process_time[job_id, :])
            feature = [
                (0.1 + machines_endtime[machine_id]) / (0.1 + np.average(machines_endtime)),
                (0.1 + job_cur_endtime) / (0.1 + job_overall_time)
            ]
        return feature

    def get_features(self, progresses):
        features = []
        for job_id, job_progress in progresses:
            features.append(self.get_feature(job_id, job_progress))
        return features

    def measure_act(self, actions):
        """
        Update self.info and calculate the makespan.
        """
        machines_timeline = np.zeros(self.machine_num, dtype=int)
        job_idx_in_machines = np.zeros(self.machine_num, dtype=int)
        jobs_timeline = np.zeros(self.job_num, dtype=int)

        for job_id, job_progress in actions:
            machine_id = self.jobs_process_order[job_id, job_progress]
            cur_starttime = max(machines_timeline[machine_id], jobs_timeline[job_id])
            cur_endtime = cur_starttime + self.jobs_process_time[job_id, job_progress]

            # update
            machines_timeline[machine_id] = cur_endtime
            jobs_timeline[job_id] = cur_endtime

            cur_idx = job_idx_in_machines[machine_id]
            self.starttime[machine_id, cur_idx] = cur_starttime
            self.endtime[machine_id, cur_idx] = cur_endtime
            self.scheduling_plan[machine_id, cur_idx, :] = [job_id, job_progress]
            job_idx_in_machines[machine_id] += 1

        return np.max(self.endtime)

    def step(self, action=None):
        """
        Give input action, return reward, new state, and judge whether is done
        :param action: a chosen job's id
        :return:
        """
        done = False
        if action is None:
            # env initialize
            self.measure_act(self.actions)
            state = np.array(self.get_features(self.get_jobs_progress()))
            score = 0
        else:
            job_progresses = [0 for _ in range(self.job_num)]
            for job_id, job_progress in self.actions:
                if job_progress < self.machine_num - 1:
                    # goto the next module (of this job)
                    job_progresses[job_id] = job_progress + 1
                else:
                    # job finished
                    job_progresses[job_id] = -1
            if job_progresses[action] == -1:
                # has to choose a new job, which means this action is not a good action
                done = True
                can_choose = [[i, job_progresses[i]] for i in range(self.job_num) if job_progresses[i] != -1]
                # if can_choose is not None?
                action = can_choose[0]
            else:
                action = [action, job_progresses[action]]

            self.actions.append(action)
            self.measure_act(self.actions)
            score = np.max(self.endtime)
            state = np.array(self.get_features(self.get_jobs_progress()))
        state = [np.reshape(state[i], (1, 2,)) for i in range(self.job_num)]
        return state, score, done

    def plot_scheduling_result(self, save_path):
        """
        Plot the Gatt map for the scheduling result.
        """
        tools.use_svg_display()
        colorbox = ['yellow', 'whitesmoke', 'lightyellow', 'khaki', 'silver', 'pink', 'lightgreen', 'orange', 'grey', 'r', 'brown']
        for i in range(100):
            color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
            color = ''
            for _ in range(6):
                color += color_arr[random.randint(0, 14)]
            colorbox.append('#' + color)

        fig = plt.figure(figsize=(7, 4))
        for i in range(self.machine_num):
            for j in range(self.job_num):
                m_point1 = self.starttime[i, j]
                m_point2 = self.endtime[i, j]
                m_text = i + 1.5
                plot_rect(m_point1, m_point2, m_text)

                words = str(self.scheduling_plan[i, j, 0] + 1) + '.' + str(str(self.scheduling_plan[i, j, 1] + 1))
                x1, x2, x3, x4 = m_point1, m_point2, m_point2, m_point1
                y1, y2, y3, y4 = m_text - 0.8, m_text - 0.8, m_text, m_text
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], color=colorbox[self.scheduling_plan[i, j, 0]])
                plt.text(0.5 * m_point1 + 0.5 * m_point2 - 3.5, m_text - 0.5, words)
        plt.xlabel('Time')
        plt.ylabel('Machine')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()


def plot_rect(m_point1, m_point2, m_text):
    v_point = np.zeros((4, 2))
    v_point[0, :] = [m_point1, m_text - 0.8]
    v_point[1, :] = [m_point2, m_text - 0.8]
    v_point[2, :] = [m_point1, m_text]
    v_point[3, :] = [m_point2, m_text]
    plt.plot([v_point[0, 0], v_point[1, 0]], [v_point[0, 1], v_point[1, 1]], 'k')
    plt.plot([v_point[0, 0], v_point[2, 0]], [v_point[0, 1], v_point[2, 1]], 'k')
    plt.plot([v_point[1, 0], v_point[3, 0]], [v_point[1, 1], v_point[3, 1]], 'k')
    plt.plot([v_point[2, 0], v_point[3, 0]], [v_point[2, 1], v_point[3, 1]], 'k')


if __name__ == '__main__':
    env = Env(4, 5)
