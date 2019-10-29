            from matplotlib import pyplot as plt
            import numpy as np
            import torch
            from torch.distributions import Normal

            T = int(1e5) # total time step
            alpha = 1e-4 # learning rate

            mu_history = []
            sigma_history = []

            policy = torch.nn.Sequential(
                torch.nn.Linear(1, 2, bias=False)
            )

            optimizer = torch.optim.SGD(params=policy.parameters(), lr=alpha)

            for t in range(T):
                mu, sigma = policy(torch.ones(1))
                dist = Normal(loc=mu, scale=sigma)
                action = dist.sample()
                reward = - np.square(action - 10)
                mu_history.append(mu.item())
                sigma_history.append(sigma.item())
                loss = -dist.log_prob(action) * reward # policy gradient objective
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            plt.subplot(1,2,1)
            plt.title('Estimation of mu')
            plt.plot(mu_history)
            plt.subplot(1,2,2)
            plt.title('Estimation of sigma')
            plt.plot(sigma_history)
            plt.show()




