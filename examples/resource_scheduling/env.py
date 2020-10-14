"""
Use Deep Q-Network to learn how to schedule jobs.
This file defines the state, action, and reward of this environment.

    Author: Chang Li and hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import numpy as np


WORK_ORDER_FILEPATH = '../../data/Scheduling/work_order.csv'
PROCESSING_TIME = '../../data/Scheduling/process_time_matrix.csv'
JOB_ATTR_NUM = 10
EXPERT_ATTR_NUM = 7 + 107


def load_data(work_order_file=WORK_ORDER_FILEPATH, processing_time_file=PROCESSING_TIME):
    # every index starts from 1
    with open(work_order_file, 'r', encoding='utf-8') as f:
        work_order_str = f.read()
    work_order = work_order_str.strip().split('\n')
    work_order = [o.strip().split(',') for o in work_order]
    work_order = np.array([[int(v) for v in o] for o in work_order], dtype='int32')
    work_order = np.vstack([[0, 0, 0, 0], work_order])

    with open(processing_time_file, 'r', encoding='utf-8') as f:
        processing_time_str = f.read()
    processing_time = processing_time_str.strip().split('\n')
    processing_time = [p.strip().split(',') for p in processing_time]
    processing_time = np.array([[int(v) for v in p] for p in processing_time], dtype='int32')

    return work_order, processing_time


class JobState:
    """
    Define the job state.
    """
    def __init__(self, j_id, appear_time, j_type, max_response_time):
        self.j_id = j_id
        self.appear_time = appear_time
        self.j_type = j_type
        self.max_response_time = max_response_time

        self.migration_count = 0
        self.is_start = False
        self.start_process_time = -1
        self.pre_assigned_to = -1
        self.stay_start_time = -1   # record the processing time since dispatching
        self.is_finish = False

    def __str__(self):
        return 'Job State:\n ' \
               'id: {}\n, type: {}\n appear time: {}\n max response time: {}\n' \
               'migration count: {}\n is started? {}\n start process time: {}\n previously assigned to: {}\n' \
               'stay start time: {}\n is finished? {}\n'.\
            format(self.j_id,
                   self.j_type,
                   self.appear_time,
                   self.max_response_time,
                   self.migration_count,
                   self.is_start,
                   self.start_process_time,
                   self.pre_assigned_to,
                   self.stay_start_time,
                   self.is_finish)


class ExpertState:
    """
    Define the expert state.
    """
    def __init__(self, e_id, process_time):
        self.e_id = e_id
        self.process_time = process_time

        self.queue1_is_busy = False
        self.q1_cur_process_job_id = -1
        self.queue2_is_busy = False
        self.q2_cur_process_job_id = -1
        self.queue3_is_busy = False
        self.q3_cur_process_job_id = -1

    def __str__(self):
        return 'Expert State:\n id: {}\n process_time: {}\n ' \
               'queue1 is processing? {}, who: {}\n ' \
               'queue2 is processing? {}, who: {}\n ' \
               'queue3 is processing? {}, who: {}\n'.\
            format(self.e_id, self.process_time,
                   self.queue1_is_busy, self.q1_cur_process_job_id,
                   self.queue2_is_busy, self.q2_cur_process_job_id,
                   self.queue3_is_busy, self.q3_cur_process_job_id)


class State:
    """
    Define the state for RL environment (the simple join of jobs' state and experts' state.
    """
    def __init__(self):
        job_order, expert_process_time = load_data()

        self.job_state = []
        self.job_state.append(JobState(0, 0, 0, 0))
        for i in range(1, len(job_order)):
            self.job_state.append(JobState(job_order[i][0],
                                           job_order[i][1],
                                           job_order[i][2],
                                           job_order[i][3]))

        self.expert_state = []
        self.expert_state.append(ExpertState(0, np.array([0] * 107)))
        for i in range(1, len(expert_process_time)):
            self.expert_state.append(ExpertState(expert_process_time[i][0], expert_process_time[i, 1:]))

    def convert_into_array(self):
        """
        The numpy array for state:
        state = [[job1_attr1, ..., job1_attr10, 0, ..., 0,],
                 ...   ...   ...
                 [job8840_attr1, ..., job8840_attr10, 0, ..., 0,],
                 [exp1_attr1, ..., exp1_attr114],
                 ...   ...   ...
                 [exp133_attr1, ..., exp133_attr114]]
        ```
        The size of state is (8840 + 1 + 133 + 1) * 114 = 8975 * 114.
        """
        job_attr_num = JOB_ATTR_NUM
        expert_attr_num = EXPERT_ATTR_NUM
        state_array = []
        for i in range(len(self.job_state)):
            tmp = [self.job_state[i].j_id,
                   self.job_state[i].j_type,
                   self.job_state[i].appear_time,
                   self.job_state[i].max_response_time,
                   self.job_state[i].migration_count,
                   1 if self.job_state[i].is_start else 0,
                   self.job_state[i].start_process_time,
                   self.job_state[i].pre_assigned_to,
                   self.job_state[i].stay_start_time,
                   1 if self.job_state[1].is_finish else 0]
            tmp.extend([0 for _ in range((expert_attr_num - job_attr_num))])
            state_array.append(np.array(tmp.copy()))
            del tmp

        for i in range(len(self.expert_state)):
            tmp = [self.expert_state[i].e_id,
                   1 if self.expert_state[i].queue1_is_busy else 0,
                   self.expert_state[i].q1_cur_process_job_id,
                   1 if self.expert_state[i].queue2_is_busy else 0,
                   self.expert_state[i].q2_cur_process_job_id,
                   1 if self.expert_state[i].queue3_is_busy else 0,
                   self.expert_state[i].q3_cur_process_job_id]
            tmp.extend(self.expert_state[i].process_time)
            state_array.append(np.array(tmp.copy()))
            del tmp

        return np.array(state_array)


class MDP:
    def __init__(self, state):
        self.cur_time = 0
        self.finished_job_num = 0
        self.cur_state = state

    def step(self, input_act, job_id, go_ahead=False):
        """
        For current state, with given action, define the reward and new state.
        """
        if go_ahead:
            self.cur_time += 1
        is_invalid = False

        if sum(input_act) != 1:
            raise ValueError('Dispatch to multiple experts!')
        job_agent = self.cur_state.job_state[job_id]
        if job_agent.appear_time > self.cur_time:
            is_invalid = True
            reward = -10
            # re-initialize
            self.__init__(State())
        elif input_act[-1] == 1:
            if job_agent.max_response_time + job_agent.appear_time >= self.cur_time and job_agent.migration_count != 0:
                # job timeout without the first round response
                reward = -0.5
            elif job_agent.migration_count > 0:
                assigned_to = self.cur_state.expert_state[job_agent.pre_assigned_to]
                if assigned_to.process_time[job_agent.j_type] < 999999:
                    # job has been assigned to an expert who is good at this
                    reward = 0.5
                else:
                    reward = 0.4
            else:
                # job is pending
                reward = 0.3
        else:
            # job is assigned to an expert in current time slot
            job_type = job_agent.j_type
            selected_expert = np.where(input_act == 1)[0][0] + 1
            expert_agent = self.cur_state.expert_state[selected_expert]
            if selected_expert == job_agent.pre_assigned_to:
                # continue processing on the same expert
                if self.cur_time - job_agent.stay_start_time >= expert_agent.process_time[job_type]:
                    # job finished
                    reward = 2
                    if expert_agent.q1_cur_process_job_id == job_agent.j_id:
                        expert_agent.queue1_is_busy = False
                        expert_agent.q1_cur_process_job_id = -1
                    elif expert_agent.q2_cur_process_job_id == job_agent.j_id:
                        expert_agent.queue2_is_busy = False
                        expert_agent.q2_cur_process_job_id = -1
                    else:
                        expert_agent.queue3_is_busy = False
                        expert_agent.q3_cur_process_job_id = -1
                    job_agent.is_finish = True
                    self.finished_job_num += 1
                else:
                    if expert_agent.process_time[job_type] == 999999:
                        reward = 0.4
                    else:
                        reward = 0.5
            else:
                # dispatch/migrate to another expert
                if expert_agent.process_time[job_type] == 999999:
                    reward = 0.4
                else:
                    reward = 0.5
                if not expert_agent.queue1_is_busy:
                    expert_agent.queue1_is_busy = True
                    expert_agent.q2_cur_process_job_id = job_id
                elif not expert_agent.queue2_is_busy:
                    expert_agent.queue2_is_busy = True
                    expert_agent.q2_cur_process_job_id = job_id
                else:
                    expert_agent.queue3_is_busy = True
                    expert_agent.q3_cur_process_job_id = job_id

                job_agent.migration_count += 1
                if not job_agent.is_start:
                    # dispatch to an expert for the first time
                    job_agent.is_start = True
                job_agent.start_process_time = self.cur_time
                job_agent.pre_assigned_to = expert_agent.e_id
                job_agent.stay_start_time = self.cur_time

        is_terminal = True if (self.finished_job_num == len(self.cur_state.job_state) - 1) else False
        return self.cur_state.convert_into_array(), reward, is_terminal, is_invalid


if __name__ == '__main__':
    s = State()
    print(s.job_state, s.expert_state)
