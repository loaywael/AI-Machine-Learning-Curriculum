import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        x_dsit_loss = abs(self.sim.pose[0])  # unwanted change in x-axis
        y_dist_loss = abs(self.sim.pose[1])  # unwanted change in y-axis
        # error between desired altitude and simulation altitude
        z_dist_loss = abs(self.sim.pose[2] - self.target_pos[2])  
        ang_pos_loss = (np.abs(self.sim.pose[3:])).sum()  # no change in angles should be permitted while taking-off
        vel_loss = (np.abs(self.sim.v[:2])).sum()  # velociy change in x, y axes not permitted while taking-off
        ang_vel_loss = (np.abs(self.sim.v[3:])).sum()  # angular velocities change not permitted in vertical hover
        # total loss of all undesired status combined one value for penalty
        tot_loss = x_dsit_loss + y_dist_loss + z_dist_loss + ang_pos_loss + vel_loss + ang_vel_loss

        # discrete reward based on target loss error in z-axis
        if z_dist_loss < 1 >= 0 :  
            reward = np.tanh(1 - 0.001 * tot_loss) + 0.05  # highest reward comes with lowest error
        elif z_dist_loss < 5 >= 1: 
            reward = np.tanh(1 - 0.005 * tot_loss) + 0.01  # less reward value for increased error
        elif z_dist_loss < 9 >= 5: 
            reward = np.tanh(1 - 0.01 * tot_loss) + 0.005  # increasing the effect of penalty & decrease reward
        else:
            reward = np.tanh(1 - 0.05 * tot_loss)  # increasing penalty effect
            
        if self.sim.time < self.sim.runtime and self.sim.done:
            reward -= 1   # high penalty for crashing before runtime end
            
        return reward
    

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Older works of tunning and optimizing Reward Function
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Approach <a> old trial
#         reward = 1.-0.1*(abs(self.sim.pose[:3] - self.target_pos)).sum()
#         reward = 0
#         delta_x = abs(self.sim.pose[0] - self.target_pos[0])
#         delta_y = abs(self.sim.pose[1] - self.target_pos[1])
#         delta_z = abs(self.sim.pose[2] - self.target_pos[2])
#         penalty = 0.3 * np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
#         reward += 5 - penalty
#         if reward > 1:
#             reward = 1
#         if reward < -1:
#             reward = -1
#         return reward


#         reward = np.tanh(1.-0.0005*(abs(self.sim.pose[2] - self.target_pos[2])))

#         if self.sim.done and self.sim.time < self.sim.runtime:
#             reward = -1
#         # clipping the reward between (-1, 1) to avoid instability in training due to exploding gradients. 
#         return np.clip(reward, -1, 1)


        # Approach <b> old trial
#         mse = 1/3 * (delta_x + delta_y + delta_z)**2
#         reward += 3 - mse
#         return np.tanh(reward)

#         reward = 0
#         # penalizing the difference in vertical distance hence the task in hover "take-off"
#         penalty =  -0.005*np.tanh(np.sqrt((self.sim.pose[2] - self.target_pos[2])**2))
#         # gradually increasing the take off speed would help increase reward i.e decrease error faster
#         reward += 2 + (0.07 * self.sim.v[2]) + penalty
#         # if the drone crashed before the simulation time ends it's a high penalty
#         if self.sim.done and self.sim.time < self.sim.runtime:
#             reward = -5
#         # clipping the reward between (-1, 1) to avoid instability in training due to exploding gradients. 
#         return np.clip(reward, -1, 1)

    
    
  
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state