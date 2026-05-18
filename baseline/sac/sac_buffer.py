import random
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self,
             state, action, reward, next_state, done,
             yaw_err, v_err,
             next_yaw_err, next_v_err):
        data = (
            state.detach().cpu(),
            action.detach().cpu(),
            float(reward),
            next_state.detach().cpu(),
            float(done),
            yaw_err.detach().cpu(),
            v_err.detach().cpu(),
            next_yaw_err.detach().cpu(),
            next_v_err.detach().cpu(),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        (state, action, reward, next_state, done,
         yaw_err, v_err, next_yaw_err, next_v_err) = zip(*batch)

        state = torch.stack(state).to(device)
        action = torch.stack(action).to(device)
        reward = torch.tensor(reward, dtype=torch.float32,
                              device=device).unsqueeze(-1)
        next_state = torch.stack(next_state).to(device)
        done = torch.tensor(done, dtype=torch.float32,
                            device=device).unsqueeze(-1)

        yaw_err = torch.stack(yaw_err).to(device)
        v_err = torch.stack(v_err).to(device)
        next_yaw_err = torch.stack(next_yaw_err).to(device)
        next_v_err = torch.stack(next_v_err).to(device)

        return (state, action, reward, next_state, done,
                yaw_err, v_err, next_yaw_err, next_v_err)

    def __len__(self):
        return len(self.buffer)
